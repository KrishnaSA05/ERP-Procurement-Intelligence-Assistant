"""
src/agents/bedrock_client.py
─────────────────────────────
Shared Amazon Bedrock client used by all agents.
Returns a LangChain-compatible ChatBedrockConverse instance.

Single place to configure:
  • model ID
  • temperature / max_tokens
  • retry logic
  • region

Usage:
    from src.agents.bedrock_client import get_llm
    llm = get_llm(temperature=0)
"""

import os
from functools import lru_cache

from langchain_aws import ChatBedrockConverse
from loguru import logger


# Default model — Claude Haiku (fast, cheap, capable enough for SQL + RAG)
DEFAULT_MODEL = "anthropic.claude-haiku-20240307-v1:0"


@lru_cache(maxsize=4)
def get_llm(
    temperature : float = 0.0,
    max_tokens  : int   = 2048,
    model_id    : str   = None,
) -> ChatBedrockConverse:
    """
    Returns a cached LangChain ChatBedrockConverse instance.

    Args:
        temperature : 0.0 for deterministic SQL/classification, 0.3 for synthesis
        max_tokens  : Max output tokens
        model_id    : Override model. Defaults to Claude Haiku.

    Returns:
        ChatBedrockConverse instance (LangChain-compatible)
    """
    model   = model_id or os.getenv("BEDROCK_MODEL_ID", DEFAULT_MODEL)
    region  = os.getenv("AWS_REGION", "us-east-1")

    logger.info(f"Initialising Bedrock LLM: {model} (temp={temperature}, region={region})")

    llm = ChatBedrockConverse(
        model           = model,
        region_name     = region,
        temperature     = temperature,
        max_tokens      = max_tokens,
    )

    logger.success(f"  ✓ Bedrock LLM ready [{model}]")
    return llm
