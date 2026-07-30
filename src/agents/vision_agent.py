"""
src/agents/vision_agent.py

Agent 4 — Vision Agent.

Extracts structured procurement data (vendor, PO number, line items,
total, date) from an uploaded invoice/PO image using a VLM, so it can be
folded into the normal SQL/RAG/Hybrid pipeline as extra context.

This agent does NOT decide routing itself. Its output is merged into
state["question"] by vision_node (src/graph/nodes.py) BEFORE the existing
guardrail/classify nodes run — so the rest of the graph (guardrails,
classifier, SQL agent, RAG agent, synthesis agent) needs zero changes.
A question like "does this match our records?" plus an attached invoice
photo becomes, by the time it reaches the classifier, a single enriched
question that already contains the extracted PO number/vendor/total —
which is exactly the kind of thing that naturally routes to "hybrid"
(look up the PO in Postgres, check the contract for matching terms).

Usage:
    from src.agents.vision_agent import VisionAgent
    agent  = VisionAgent()
    result = agent.run(image_bytes)
    print(result.to_dict())
"""

import json
import re
from dataclasses import dataclass, field

from loguru import logger

from src.agents.vlm_client import describe_image


# ── Prompt ────────────────────────────────────────────────────────────────────

VISION_EXTRACTION_PROMPT = """You are reading a photo or scan of a procurement document (invoice or purchase order).

Extract the following fields as a JSON object, using null for anything not visible:
  - "document_type": "invoice" | "purchase_order" | "unknown"
  - "vendor_name": string or null
  - "po_number": string or null
  - "invoice_number": string or null
  - "invoice_date": string or null (exactly as written on the document)
  - "line_items": list of {"description": string, "quantity": number|null, "unit_price": number|null, "total": number|null}
  - "total_amount": number or null

Respond with ONLY the JSON object — no markdown fences, no commentary."""


# ── Result model ──────────────────────────────────────────────────────────────

@dataclass
class VisionAgentResult:
    success          : bool  = True
    error            : str   = ""
    document_type    : str   = "unknown"
    vendor_name      : str   = ""
    po_number        : str   = ""
    invoice_number   : str   = ""
    invoice_date     : str   = ""
    total_amount     : float = None
    line_items       : list  = field(default_factory=list)
    raw_vlm_response : str   = ""

    def to_dict(self) -> dict:
        return {
            "success"       : self.success,
            "error"         : self.error,
            "document_type" : self.document_type,
            "vendor_name"   : self.vendor_name,
            "po_number"     : self.po_number,
            "invoice_number": self.invoice_number,
            "invoice_date"  : self.invoice_date,
            "total_amount"  : self.total_amount,
            "line_items"    : self.line_items,
        }

    def as_context_snippet(self) -> str:
        """
        Compact one-line summary folded into state["question"] by vision_node.
        Kept short and key=value style on purpose — this is prepended to the
        user's question before it reaches the classifier/SQL/RAG prompts, so
        it shouldn't blow up their context budget.

        Also surfaces the bare numeric id parsed out of "PO-00096"/"INV-00092"
        style references (a formatting convention this project's test-data
        generator uses, common on real scanned documents too) — the ERP
        schema's po_id/invoice_id columns are plain integers with no prefix,
        so handing the SQL agent an already-parsed number is more reliable
        than hoping its text-to-SQL step reliably strips the prefix itself.
        """
        if not self.success:
            return "[Vision extraction failed — no structured data available.]"

        parts = [f"document_type={self.document_type}"]
        if self.vendor_name:
            parts.append(f"vendor={self.vendor_name}")
        if self.po_number:
            parts.append(f"po_number={self.po_number}")
            po_id = _parse_reference_id(self.po_number)
            if po_id is not None:
                parts.append(f"po_id={po_id}")
        if self.invoice_number:
            parts.append(f"invoice_number={self.invoice_number}")
            invoice_id = _parse_reference_id(self.invoice_number)
            if invoice_id is not None:
                parts.append(f"invoice_id={invoice_id}")
        if self.invoice_date:
            parts.append(f"date={self.invoice_date}")
        if self.total_amount is not None:
            parts.append(f"total={self.total_amount}")
        if self.line_items:
            parts.append(f"line_items={len(self.line_items)}")

        return "[Extracted from uploaded image: " + ", ".join(parts) + "]"


# ── Reference parsing helper ───────────────────────────────────────────────────

def _parse_reference_id(reference: str):
    """
    Extracts the trailing numeric id from a formatted reference like
    "PO-00096" or "INV-00092" → 96 / 92 (leading zeros stripped naturally
    by int()). Returns None if no digits are found, rather than raising —
    this is a best-effort hint for the SQL agent, not a strict parser.
    """
    if not reference:
        return None
    match = re.search(r"(\d+)\s*$", reference.strip())
    if not match:
        return None
    return int(match.group(1))


# ── JSON parsing helper ────────────────────────────────────────────────────────

def _parse_json_response(text: str) -> dict:
    """
    VLMs occasionally wrap JSON in markdown fences or add a stray sentence
    despite instructions not to. Strip fences, then grab the outermost {...}.
    """
    cleaned = text.strip()
    cleaned = re.sub(r"^```(json)?|```$", "", cleaned, flags=re.MULTILINE).strip()

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in VLM response: {text[:200]!r}")

    return json.loads(match.group(0))


# ── Vision Agent ──────────────────────────────────────────────────────────────

class VisionAgent:
    """
    Agent 4 — extracts structured fields from an invoice/PO image via a VLM.
    """

    def __init__(self):
        logger.info("VisionAgent initialised")

    def run(self, image_bytes: bytes, question: str = "") -> VisionAgentResult:
        """
        Args:
            image_bytes : Raw bytes of the uploaded invoice/PO image.
            question    : Optional original user question (currently unused
                          in the prompt itself, accepted for API symmetry
                          with the other agents' .run(question) signature
                          and for future prompt customisation).

        Returns:
            VisionAgentResult with extracted fields (success=False + error
            set on failure — never raises, matching SQLAgent/RAGAgent).
        """
        logger.info(f"VisionAgent running on {len(image_bytes)} byte image")

        try:
            raw = describe_image(
                image_bytes = image_bytes,
                prompt      = VISION_EXTRACTION_PROMPT,
                label       = "vision_agent",
            )
            fields = _parse_json_response(raw)

            result = VisionAgentResult(
                success          = True,
                document_type    = fields.get("document_type") or "unknown",
                vendor_name      = fields.get("vendor_name") or "",
                po_number        = fields.get("po_number") or "",
                invoice_number   = fields.get("invoice_number") or "",
                invoice_date     = fields.get("invoice_date") or "",
                total_amount     = fields.get("total_amount"),
                line_items       = fields.get("line_items") or [],
                raw_vlm_response = raw,
            )
            logger.success(
                f"  VisionAgent extracted: vendor={result.vendor_name!r} "
                f"po_number={result.po_number!r} total={result.total_amount}"
            )
            return result

        except Exception as e:
            logger.error(f"  VisionAgent error: {e}")
            return VisionAgentResult(success=False, error=str(e))


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("Usage: python vision_agent.py <path_to_invoice_image>")
        sys.exit(1)

    with open(path, "rb") as f:
        image_bytes = f.read()

    agent  = VisionAgent()
    result = agent.run(image_bytes)
    print(json.dumps(result.to_dict(), indent=2))
    print(f"\nContext snippet: {result.as_context_snippet()}")
