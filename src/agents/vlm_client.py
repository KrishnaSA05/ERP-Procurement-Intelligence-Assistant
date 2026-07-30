"""
src/agents/vlm_client.py
─────────────────────────
Vision-Language Model client — auto-selects backend based on APP_ENV,
mirroring bedrock_client.py's / llm_gateway.py's text-LLM split:

  development → Groq (qwen/qwen3.6-27b) — same GROQ_API_KEY, same
                langchain_groq.ChatGroq client already used for text
  production  → Amazon Bedrock Claude (already multimodal — reuses the
                same AWS account/region as the text LLM, no new service)

Both backends are online API calls (no local model weights, no separate
inference server to run) — this mirrors exactly how the existing text
LLM calls in this repo already work.

Note: Groq serves qwen/qwen3.6-27b as a PREVIEW model — fine for a dev/
demo environment, but Groq's vision lineup has churned through several
model names in the past (llama-3.2-vision-preview → llama-4-scout →
qwen3.6-27b). If a future migration is needed, only DEFAULT_GROQ_VLM_MODEL
/ GROQ_VLM_MODEL below need to change.

Usage:
    from src.agents.vlm_client import describe_image

    raw_text = describe_image(
        image_bytes = jpg_bytes,
        prompt      = "Extract vendor, PO number, line items and total as JSON.",
        label       = "vision_agent",
    )
"""

import os
import base64

from loguru import logger


DEFAULT_GROQ_VLM_MODEL    = "qwen/qwen3.6-27b"    # Groq's current vision-capable model (preview)
DEFAULT_BEDROCK_VLM_MODEL = "anthropic.claude-3-5-sonnet-20240620-v1:0"   # vision-capable on Bedrock


# ── Helpers ───────────────────────────────────────────────────────────────────

def _b64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


# ── Dev backend: Groq (vision-capable model) ──────────────────────────────────

def _call_groq(
    image_bytes: bytes,
    prompt     : str,
    model      : str,
) -> str:
    """
    Calls Groq's vision-capable chat model via langchain_groq.ChatGroq —
    the same client class llm_gateway.py already uses for text, just with
    an image content block added to the message (Groq's API is OpenAI-
    compatible, so image_url + a base64 data URI is the standard format).
    """
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set — required for the vision (VLM) client in dev")

    llm = ChatGroq(model=model, api_key=api_key, temperature=0.0, max_tokens=1024)

    message = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {
            "type"     : "image_url",
            "image_url": {"url": f"data:image/png;base64,{_b64(image_bytes)}"},
        },
    ])
    response = llm.invoke([message])
    return response.content.strip()


# ── Prod backend: Bedrock Claude (vision-capable) ─────────────────────────────

def _call_bedrock(
    image_bytes: bytes,
    prompt     : str,
    model      : str,
    region     : str,
) -> str:
    """
    Calls Bedrock Claude with an image content block via ChatBedrockConverse.
    Claude on Bedrock is natively multimodal, so no separate vision endpoint
    or SDK is needed beyond what bedrock_client.py already uses.
    """
    from langchain_aws import ChatBedrockConverse
    from langchain_core.messages import HumanMessage

    llm = ChatBedrockConverse(
        model       = model,
        region_name = region,
        temperature = 0.0,
        max_tokens  = 1024,
    )

    message = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {
            "type"  : "image",
            "source": {
                "type"      : "base64",
                "media_type": "image/png",
                "data"      : _b64(image_bytes),
            },
        },
    ])
    response = llm.invoke([message])
    return response.content.strip()


# ── Public entry point ────────────────────────────────────────────────────────

def describe_image(
    image_bytes: bytes,
    prompt     : str,
    label      : str = "",
) -> str:
    """
    Sends an image + prompt to the environment-appropriate VLM and returns
    its raw text response. Callers are responsible for any further parsing
    (e.g. JSON extraction in vision_agent.py).

    Args:
        image_bytes : Raw image bytes (PNG/JPEG).
        prompt      : Instruction for the VLM (e.g. extraction schema, OCR request).
        label       : Optional tag for log readability (mirrors bedrock_client's `label`).

    Returns:
        Raw text response from the VLM.
    """
    env = os.getenv("APP_ENV", "development").lower()
    tag = f":{label}" if label else ""

    if env == "production":
        model  = os.getenv("BEDROCK_VLM_MODEL_ID", DEFAULT_BEDROCK_VLM_MODEL)
        region = os.getenv("AWS_REGION", "us-east-1")
        logger.info(f"[vlm{tag}] Bedrock vision call: {model}")
        return _call_bedrock(image_bytes, prompt, model, region)

    model = os.getenv("GROQ_VLM_MODEL", DEFAULT_GROQ_VLM_MODEL)
    logger.info(f"[vlm{tag}] Groq vision call: {model}")
    return _call_groq(image_bytes, prompt, model)


# ── CLI smoke test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("Usage: python vlm_client.py <path_to_image>")
        sys.exit(1)

    with open(path, "rb") as f:
        img = f.read()

    print(describe_image(img, "Describe this image in one sentence.", label="cli_test"))
