"""
src/agents/bedrock_client.py
─────────────────────────────
LLM client — auto-selects backend based on APP_ENV:
  development → Groq (free, fast)
  production  → Amazon Bedrock Claude Haiku (pay-per-use)

Every LLM instance returned by get_llm() is wrapped in a ResilientLLM
(src/agents/llm_gateway.py) — retry with backoff on transient errors,
then a single fallback-model attempt if the primary is still failing.
This is transparent to callers: every existing `llm.invoke(messages)`
call site elsewhere in the app gets this for free, no changes needed.

Usage:
    from src.agents.bedrock_client import get_llm
    llm = get_llm(temperature=0.0, max_tokens=512, label="sql_agent")
    response = llm.invoke(messages)   # retry + fallback handled internally
"""

import os
from functools import lru_cache

from loguru import logger

from src.agents.llm_gateway import ResilientLLM


DEFAULT_BEDROCK_MODEL = "anthropic.claude-3-haiku-20240307-v1:0"
DEFAULT_GROQ_MODEL    = "openai/gpt-oss-20b"   # replaces deprecated llama-3.1-8b-instant (Groq deprecated it June 2026)


@lru_cache(maxsize=8)
def get_llm(
    temperature : float = 0.0,
    max_tokens  : int   = 2048,
    model_id    : str   = None,
    label       : str   = "",
) -> ResilientLLM:
    """
    Returns a cached, gateway-wrapped LLM instance.
    Auto-selects Groq (dev) or Bedrock (prod) based on APP_ENV.

    Args:
        temperature : 0.0 for deterministic SQL/classification, higher for synthesis
        max_tokens  : Max output tokens
        model_id    : Override model. Defaults per-backend.
        label       : Optional tag included in gateway retry/fallback logs
                      (e.g. "guardrail", "sql_agent") — purely for readability
                      when tracing which caller triggered a fallback.

    Returns:
        ResilientLLM wrapping a LangChain-compatible chat model.
    """
    env = os.getenv("APP_ENV", "development").lower()

    if env == "production":
        from langchain_aws import ChatBedrockConverse
        model  = model_id or os.getenv("BEDROCK_MODEL_ID", DEFAULT_BEDROCK_MODEL)
        region = os.getenv("AWS_REGION", "us-east-1")
        logger.info(f"Initialising Bedrock LLM: {model} (temp={temperature}, region={region})")
        llm = ChatBedrockConverse(
            model       = model,
            region_name = region,
            temperature = temperature,
            max_tokens  = max_tokens,
        )
        logger.success(f"  ✓ Bedrock LLM ready [{model}]")

    else:
        from langchain_groq import ChatGroq
        model   = model_id or os.getenv("GROQ_MODEL_ID", DEFAULT_GROQ_MODEL)
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in .env")
        logger.info(f"Initialising Groq LLM: {model} (temp={temperature})")
        llm = ChatGroq(
            model       = model,
            api_key     = api_key,
            temperature = temperature,
            max_tokens  = max_tokens,
        )
        logger.success(f"  ✓ Groq LLM ready [{model}]")

    return ResilientLLM(llm, temperature=temperature, max_tokens=max_tokens, label=label)
