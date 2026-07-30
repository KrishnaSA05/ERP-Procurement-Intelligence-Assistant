"""
src/ingestion/vlm_ocr.py
──────────────────────────
Fallback OCR for scanned/image-only PDF pages using the VLM client.

pdf_loader.py's pypdf/pdfplumber extraction returns near-empty text for
scanned pages — signature pages, stamped PO approvals, image-only contract
addenda. Previously those pages were silently dropped (see
`min_chars_per_page` in pdf_loader.py). This module rasterizes such a page
to an image and asks the VLM to transcribe it, so its content still makes
it into the chunker/embedder/ChromaDB pipeline like any other page.

Usage:
    from src.ingestion.vlm_ocr import rasterize_page, vlm_ocr_page

    png_bytes = rasterize_page(pdf_path, page_number=3)   # 1-indexed
    text      = vlm_ocr_page(png_bytes)
"""

from pathlib import Path

import fitz  # PyMuPDF
from loguru import logger

from src.agents.vlm_client import describe_image


OCR_PROMPT = """Transcribe ALL visible text from this scanned document page, exactly as written,
preserving line breaks and structure where reasonable. For stamps or signatures,
write a short bracketed placeholder instead of guessing illegible marks — e.g.
"[signature]" or "[stamp: Approved]". Include table contents as plain text rows.
Respond with ONLY the transcribed text — no commentary, no markdown fences."""


def rasterize_page(pdf_path: str | Path, page_number: int, dpi: int = 200) -> bytes:
    """
    Renders a single PDF page to PNG bytes using PyMuPDF.

    Args:
        pdf_path    : Path to the PDF file.
        page_number : 1-indexed page number (matches PDFPage.page_number).
        dpi         : Render resolution — 200 is a reasonable OCR/quality tradeoff.

    Returns:
        PNG-encoded image bytes for that page.
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_number - 1]
        zoom = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix)
        return pix.tobytes("png")
    finally:
        doc.close()


def vlm_ocr_page(image_bytes: bytes) -> str:
    """
    Sends a rasterized page image to the VLM and returns transcribed text.
    Never raises — callers (pdf_loader.py) treat an empty string the same
    as "OCR yielded nothing usable" and fall back to skipping the page,
    same as the original behaviour.
    """
    try:
        text = describe_image(image_bytes=image_bytes, prompt=OCR_PROMPT, label="vlm_ocr")
        return text.strip()
    except Exception as e:
        logger.warning(f"  [vlm_ocr] VLM transcription failed: {e}")
        return ""
