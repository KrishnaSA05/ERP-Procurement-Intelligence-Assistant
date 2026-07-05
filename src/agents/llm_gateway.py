"""
src/agents/llm_gateway.py
LLM Gateway — wraps the raw LangChain LLM client with retry + fallback
resilience.

Every agent in this app (guardrails, classifier, SQL agent, RAG agent,
synthesis agent) gets its LLM instance from bedrock_client.get_llm().
get_llm() now returns a ResilientLLM instead of a raw client — so this
gateway sits transparently in front of every LLM call in the system
without any call site anywhere else needing to change. Every existing
`llm.invoke(messages)` call automatically gets retry + fallback for free.

Resilience strategy:
  1. RETRY    — transient errors (rate limits, timeouts, connection drops)
               on the PRIMARY backend get retried with exponential backoff
               (tenacity), up to LLM_GATEWAY_MAX_RETRIES attempts.
  2. FALLBACK — if the primary is still failing after retries, ONE attempt
               is made against a separate fallback model:
                 development → Groq primary model    -> Groq fallback model
                 production  → Bedrock primary model  -> Bedrock fallback model
  3. RAISE    — if both primary and fallback are exhausted, the original
               exception propagates. Every agent node in nodes.py already
               wraps its work in try/except and records failures into
               state["errors"] — so a fully-exhausted gateway degrades
               that one node gracefully rather than crashing the request.

This mirrors the primary/fallback pattern in the 8hr-MARATHON repo's
Portkey gateway config, but implemented natively — tenacity is already
in requirements.txt, and nothing else in this repo depends on a gateway
SDK, so a native wrapper stays consistent with the rest of the codebase
(same reasoning as guardrails.py: build it, don't bolt on a new dependency
for something this codebase's own tools can already do).
"""

import os

from loguru import logger
from tenacity import (
    retry, stop_after_attempt, wait_exponential, retry_if_exception,
)


LLM_GATEWAY_MAX_RETRIES = int(os.getenv("LLM_GATEWAY_MAX_RETRIES", "2"))


# ── Transient-error detection ────────────────────────────────────────────────
# Checks known SDK-specific exception types where importable, plus a
# keyword fallback so this stays resilient to SDK version differences
# instead of silently never retrying if an exact type isn't matched.

_TRANSIENT_KEYWORDS = (
    "rate limit", "ratelimit", "throttl", "429", "503", "timeout",
    "timed out", "connection", "service unavailable", "overloaded",
)


def _is_transient(exc: BaseException) -> bool:
    try:
        from groq import RateLimitError, APITimeoutError, APIConnectionError
        if isinstance(exc, (RateLimitError, APITimeoutError, APIConnectionError)):
            return True
    except ImportError:
        pass

    try:
        from botocore.exceptions import ClientError
        if isinstance(exc, ClientError):
            code = exc.response.get("Error", {}).get("Code", "")
            if code in (
                "ThrottlingException", "ServiceUnavailableException",
                "ModelTimeoutException", "TooManyRequestsException",
            ):
                return True
    except ImportError:
        pass

    msg = str(exc).lower()
    return any(kw in msg for kw in _TRANSIENT_KEYWORDS)


def _log_retry(retry_state):
    exc = retry_state.outcome.exception()
    logger.warning(
        f"  [gateway] Transient error on attempt {retry_state.attempt_number}: "
        f"{exc}. Retrying..."
    )


# ── Fallback model config ────────────────────────────────────────────────────

DEFAULT_GROQ_FALLBACK_MODEL    = "openai/gpt-oss-120b"   # different capacity tier from primary (20b)
DEFAULT_BEDROCK_FALLBACK_MODEL = "anthropic.claude-3-5-sonnet-20240620-v1:0"   # different tier from Haiku


def _build_fallback_client(temperature: float, max_tokens: int):
    """
    Builds a fallback LLM client, lazily and only if the primary has
    actually failed. Lazy-imports the SDK so a dev box without AWS
    reachability isn't forced to have Bedrock available just because
    this module got imported.
    """
    env = os.getenv("APP_ENV", "development").lower()

    if env == "production":
        from langchain_aws import ChatBedrockConverse
        model  = os.getenv("BEDROCK_FALLBACK_MODEL_ID", DEFAULT_BEDROCK_FALLBACK_MODEL)
        region = os.getenv("AWS_REGION", "us-east-1")
        logger.info(f"  [gateway] Building Bedrock FALLBACK client: {model}")
        return ChatBedrockConverse(
            model=model, region_name=region,
            temperature=temperature, max_tokens=max_tokens,
        )

    from langchain_groq import ChatGroq
    model   = os.getenv("GROQ_FALLBACK_MODEL_ID", DEFAULT_GROQ_FALLBACK_MODEL)
    # Allow a separate fallback API key (mirrors 8hr-MARATHON's primary/secondary
    # key pattern — useful if the primary key hits an account-level rate limit
    # rather than a per-model one). Falls back to the same key if not set.
    api_key = os.getenv("GROQ_FALLBACK_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("No GROQ_API_KEY/GROQ_FALLBACK_API_KEY set for fallback client")
    logger.info(f"  [gateway] Building Groq FALLBACK client: {model}")
    return ChatGroq(
        model=model, api_key=api_key,
        temperature=temperature, max_tokens=max_tokens,
    )


# ── The resilient wrapper ────────────────────────────────────────────────────

class ResilientLLM:
    """
    Drop-in replacement for a raw LangChain chat model.
    Wraps .invoke() with retry (on the primary) + a single fallback attempt.

    Unknown attributes (e.g. .bind(), .stream()) delegate straight to the
    wrapped primary client, so this stays a safe drop-in even if a call
    site starts using methods beyond .invoke() later.
    """

    def __init__(self, primary, temperature: float, max_tokens: int, label: str = ""):
        self._primary     = primary
        self._temperature = temperature
        self._max_tokens  = max_tokens
        self._label       = label     # e.g. "guardrail" / "sql_agent" — log context only
        self._fallback    = None      # built lazily, only if ever needed

    def __getattr__(self, name):
        return getattr(self._primary, name)

    def _tag(self) -> str:
        return f":{self._label}" if self._label else ""

    def invoke(self, messages, *args, **kwargs):
        try:
            return self._invoke_with_retry(self._primary, messages, *args, **kwargs)
        except Exception as primary_exc:
            logger.error(
                f"  [gateway{self._tag()}] Primary backend exhausted retries: "
                f"{primary_exc}. Attempting fallback model..."
            )
            try:
                if self._fallback is None:
                    self._fallback = _build_fallback_client(
                        self._temperature, self._max_tokens
                    )
                result = self._fallback.invoke(messages, *args, **kwargs)
                logger.success(
                    f"  [gateway{self._tag()}] Fallback model served the request."
                )
                return result
            except Exception as fallback_exc:
                logger.critical(
                    f"  [gateway{self._tag()}] Fallback ALSO failed: "
                    f"{fallback_exc}. Raising original error."
                )
                raise primary_exc

    def _invoke_with_retry(self, client, messages, *args, **kwargs):
        @retry(
            stop=stop_after_attempt(LLM_GATEWAY_MAX_RETRIES),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception(_is_transient),
            before_sleep=_log_retry,
            reraise=True,
        )
        def _call():
            return client.invoke(messages, *args, **kwargs)

        return _call()
