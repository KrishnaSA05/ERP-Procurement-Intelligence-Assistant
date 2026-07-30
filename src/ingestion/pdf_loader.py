"""
src/ingestion/pdf_loader.py
────────────────────────────
Loads PDF documents from the local contracts/ and policies/ directories.
Extracts text page-by-page and attaches rich metadata for retrieval.

Supports two backends (auto-selected by file type / quality):
  • pypdf       — fast, good for text-native PDFs
  • pdfplumber  — better for complex layouts, tables, headers

Usage:
    from src.ingestion.pdf_loader import load_documents
    docs = load_documents("data/contracts")
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Literal

import pypdf
import pdfplumber
from loguru import logger


def _vlm_ocr_enabled() -> bool:
    """
    Controls whether scanned/low-text pages get a VLM-OCR rescue attempt
    (see vlm_ocr_fallback below) instead of being silently dropped like
    the original behaviour. Defaults on; set VLM_OCR_ENABLED=false in .env
    to restore the original skip-only behaviour (e.g. no Groq/Bedrock
    reachable in this environment).
    """
    return os.getenv("VLM_OCR_ENABLED", "true").lower() in ("1", "true", "yes")


def _vlm_ocr_fallback(path: Path, page_number: int) -> str:
    """
    Lazily imports src.ingestion.vlm_ocr (which itself lazily reaches the
    VLM client) so a dev box with VLM_OCR_ENABLED=false never needs
    PyMuPDF/Groq/Bedrock reachable just because pdf_loader was imported.
    """
    from src.ingestion.vlm_ocr import rasterize_page, vlm_ocr_page
    image_bytes = rasterize_page(path, page_number)
    return vlm_ocr_page(image_bytes)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class PDFPage:
    """A single extracted page with full metadata."""
    text        : str
    page_number : int
    total_pages : int
    source_file : str          # filename only  e.g. "vendor_contract_techcorp.pdf"
    source_path : str          # full path
    doc_type    : str          # "contract" | "policy"
    doc_id      : str          # slug derived from filename  e.g. "vendor_contract_techcorp"
    extraction_method : str = "text"   # "text" (pypdf/pdfplumber) | "vlm_ocr" (VLM rescue)


@dataclass
class PDFDocument:
    """All pages extracted from a single PDF file."""
    doc_id      : str
    source_file : str
    source_path : str
    doc_type    : str
    total_pages : int
    pages       : list[PDFPage] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text.strip())


# ── Helpers ───────────────────────────────────────────────────────────────────

def _doc_type_from_path(path: Path) -> str:
    """Infer document type from folder name."""
    parts = [p.lower() for p in path.parts]
    if "contracts" in parts:
        return "contract"
    if "policies" in parts or "policy" in parts:
        return "policy"
    return "document"


def _doc_id_from_filename(filename: str) -> str:
    """Convert filename to a clean slug: 'Vendor Contract TechCorp.pdf' → 'vendor_contract_techcorp'"""
    stem = Path(filename).stem
    return stem.lower().replace(" ", "_").replace("-", "_")


def _extract_with_pypdf(path: Path) -> list[str]:
    """Extract text page-by-page using pypdf."""
    texts = []
    with open(path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            texts.append(page.extract_text() or "")
    return texts


def _extract_with_pdfplumber(path: Path) -> list[str]:
    """Extract text page-by-page using pdfplumber (better for complex layouts)."""
    texts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            texts.append(page.extract_text() or "")
    return texts


def _choose_backend(path: Path, prefer: Literal["pypdf", "pdfplumber", "auto"] = "auto") -> str:
    """Select extraction backend. 'auto' picks pdfplumber for large files."""
    if prefer != "auto":
        return prefer
    size_mb = path.stat().st_size / (1024 * 1024)
    return "pdfplumber" if size_mb > 2 else "pypdf"


# ── Core loader ───────────────────────────────────────────────────────────────

def load_pdf(
    path: str | Path,
    backend: Literal["pypdf", "pdfplumber", "auto"] = "auto",
    min_chars_per_page: int = 50,
) -> PDFDocument:
    """
    Load a single PDF file into a PDFDocument with per-page text + metadata.

    Args:
        path               : Path to the PDF file.
        backend            : Extraction backend. 'auto' selects based on file size.
        min_chars_per_page : Pages with fewer characters are skipped (likely scanned/blank).

    Returns:
        PDFDocument with pages list populated.
    """
    path      = Path(path)
    doc_type  = _doc_type_from_path(path)
    doc_id    = _doc_id_from_filename(path.name)
    chosen_be = _choose_backend(path, backend)

    logger.info(f"Loading '{path.name}' [{chosen_be}]")

    try:
        if chosen_be == "pdfplumber":
            page_texts = _extract_with_pdfplumber(path)
        else:
            page_texts = _extract_with_pypdf(path)
    except Exception as e:
        logger.warning(f"  {chosen_be} failed ({e}), retrying with pdfplumber...")
        page_texts = _extract_with_pdfplumber(path)

    total_pages = len(page_texts)
    pages = []

    for i, raw_text in enumerate(page_texts):
        text = raw_text.strip()
        extraction_method = "text"

        if len(text) < min_chars_per_page:
            # Likely a scanned/image-only page (signature page, stamped
            # approval, scanned addendum). Try a VLM-OCR rescue before
            # giving up on it, instead of silently dropping it.
            if _vlm_ocr_enabled():
                logger.debug(
                    f"  Page {i+1} has only {len(text)} chars — attempting VLM OCR rescue"
                )
                try:
                    ocr_text = _vlm_ocr_fallback(path, i + 1)
                except Exception as e:
                    logger.warning(f"  VLM OCR rescue failed for page {i+1}: {e}")
                    ocr_text = ""

                if len(ocr_text.strip()) >= min_chars_per_page:
                    text = ocr_text.strip()
                    extraction_method = "vlm_ocr"
                    logger.info(f"  ✓ Page {i+1} rescued via VLM OCR ({len(text)} chars)")
                else:
                    logger.debug(f"  Skipping page {i+1} — VLM OCR also yielded too little text")
                    continue
            else:
                logger.debug(f"  Skipping page {i+1} (only {len(text)} chars, VLM OCR disabled)")
                continue

        pages.append(PDFPage(
            text        = text,
            page_number = i + 1,
            total_pages = total_pages,
            source_file = path.name,
            source_path = str(path),
            doc_type    = doc_type,
            doc_id      = doc_id,
            extraction_method = extraction_method,
        ))

    rescued = sum(1 for p in pages if p.extraction_method == "vlm_ocr")
    logger.success(
        f"  ✓ '{path.name}' — {len(pages)}/{total_pages} pages extracted"
        + (f" ({rescued} via VLM OCR)" if rescued else "")
    )

    return PDFDocument(
        doc_id      = doc_id,
        source_file = path.name,
        source_path = str(path),
        doc_type    = doc_type,
        total_pages = total_pages,
        pages       = pages,
    )


def load_documents(
    directory: str | Path,
    recursive: bool = False,
    backend: Literal["pypdf", "pdfplumber", "auto"] = "auto",
) -> list[PDFDocument]:
    """
    Load all PDF files from a directory.

    Args:
        directory : Path to folder containing PDFs.
        recursive : If True, also searches sub-folders.
        backend   : Extraction backend per file.

    Returns:
        List of PDFDocument objects, one per file.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    pattern = "**/*.pdf" if recursive else "*.pdf"
    pdf_files = sorted(directory.glob(pattern))

    if not pdf_files:
        logger.warning(f"No PDF files found in: {directory}")
        return []

    logger.info(f"Found {len(pdf_files)} PDFs in '{directory}'")
    documents = []

    for pdf_path in pdf_files:
        try:
            doc = load_pdf(pdf_path, backend=backend)
            documents.append(doc)
        except Exception as e:
            logger.error(f"  Failed to load '{pdf_path.name}': {e}")
            continue

    logger.success(
        f"Loaded {len(documents)} documents, "
        f"{sum(len(d.pages) for d in documents)} pages total"
    )
    return documents


# ── CLI helper ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else "data/contracts"

    docs = load_documents(folder, recursive=True)
    print(f"\n{'='*50}")
    print(f"LOADED {len(docs)} DOCUMENTS")
    print(f"{'='*50}")
    for doc in docs:
        print(f"  [{doc.doc_type}] {doc.source_file}")
        print(f"    pages: {doc.total_pages}  |  extracted: {len(doc.pages)}")
        print(f"    preview: {doc.pages[0].text[:120].strip()}..." if doc.pages else "    (empty)")
        print()
