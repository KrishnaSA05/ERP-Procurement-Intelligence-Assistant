"""
src/ingestion/chunker.py
─────────────────────────
Splits PDFDocument pages into overlapping text chunks suitable for embedding.

Strategy: RecursiveCharacterTextSplitter
  — tries to split on paragraph → sentence → word boundaries in order
  — preserves context via overlap
  — attaches full metadata to each chunk for cited retrieval

Chunk sizes tested (see notebooks/02_contract_chunking_comparison.ipynb):
  | chunk_size | overlap | Faithfulness | Precision | Notes               |
  |------------|---------|--------------|-----------|---------------------|
  |    512     |   64    |    0.81      |  0.78     | too granular        |
  |    800     |  150    |    0.89      |  0.85     | ← selected          |
  |   1200     |  200    |    0.86      |  0.79     | loses clause context|

Usage:
    from src.ingestion.chunker import chunk_documents
    chunks = chunk_documents(documents)
"""

import re
import hashlib
from dataclasses import dataclass
from loguru import logger

from src.ingestion.pdf_loader import PDFDocument, PDFPage


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class TextChunk:
    """
    A single chunk ready for embedding and ChromaDB insertion.
    All fields map directly to ChromaDB document + metadata schema.
    """
    chunk_id    : str    # stable hash ID: sha256(doc_id + chunk_index)[:16]
    text        : str    # the actual chunk content
    doc_id      : str    # parent document slug
    source_file : str    # original PDF filename
    doc_type    : str    # "contract" | "policy"
    page_number : int    # page the chunk originated from
    chunk_index : int    # position within this document (0-based)
    total_chunks: int    # total chunks in this document (set after chunking)


# ── Splitter ──────────────────────────────────────────────────────────────────

class RecursiveChunker:
    """
    Splits text using a hierarchy of separators — preserves semantic boundaries
    better than fixed-size character splits.

    Separator priority:
      1. Double newline (paragraph break)
      2. Single newline
      3. Period + space (sentence end)
      4. Comma + space (clause end)
      5. Space (word boundary — last resort)
    """

    SEPARATORS = ["\n\n", "\n", ". ", ", ", " "]

    def __init__(self, chunk_size: int = 800, overlap: int = 150):
        self.chunk_size = chunk_size
        self.overlap    = overlap

    def split_text(self, text: str) -> list[str]:
        """Split a single string into overlapping chunks."""
        chunks = self._recursive_split(text, self.SEPARATORS)
        return self._apply_overlap(chunks)

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split on separators, working down the hierarchy."""
        if len(text) <= self.chunk_size:
            return [text]

        sep = separators[0] if separators else " "
        parts = text.split(sep)

        chunks   = []
        current  = ""

        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current.strip())
                # Part itself is too long → recurse with next separator
                if len(part) > self.chunk_size and len(separators) > 1:
                    sub_chunks = self._recursive_split(part, separators[1:])
                    chunks.extend(sub_chunks[:-1])
                    current = sub_chunks[-1] if sub_chunks else part
                else:
                    current = part

        if current.strip():
            chunks.append(current.strip())

        return [c for c in chunks if c.strip()]

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """
        Add overlap between consecutive chunks so context isn't lost at boundaries.
        Each chunk gets the tail of the previous chunk prepended.
        """
        if len(chunks) <= 1:
            return chunks

        result = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-self.overlap:].strip()
            result.append(prev_tail + " " + chunks[i])

        return result


# ── ID generation ─────────────────────────────────────────────────────────────

def _make_chunk_id(doc_id: str, chunk_index: int) -> str:
    """Generate a stable, deterministic ID for a chunk."""
    raw = f"{doc_id}__chunk_{chunk_index:04d}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ── Main chunking function ────────────────────────────────────────────────────

def chunk_documents(
    documents: list[PDFDocument],
    chunk_size: int = 800,
    overlap   : int = 150,
) -> list[TextChunk]:
    """
    Chunk all pages from a list of PDFDocuments into TextChunks.

    Args:
        documents  : Output from pdf_loader.load_documents()
        chunk_size : Target characters per chunk (default 800)
        overlap    : Characters of overlap between consecutive chunks (default 150)

    Returns:
        Flat list of TextChunk objects ready for embedding.
    """
    splitter    = RecursiveChunker(chunk_size=chunk_size, overlap=overlap)
    all_chunks  : list[TextChunk] = []

    for doc in documents:
        doc_chunks_raw = []

        for page in doc.pages:
            # Clean text: collapse excess whitespace, fix encoding artifacts
            clean = _clean_text(page.text)
            if not clean:
                continue

            splits = splitter.split_text(clean)
            for split in splits:
                doc_chunks_raw.append((split, page.page_number))

        total = len(doc_chunks_raw)

        for idx, (text, page_num) in enumerate(doc_chunks_raw):
            chunk = TextChunk(
                chunk_id    = _make_chunk_id(doc.doc_id, idx),
                text        = text,
                doc_id      = doc.doc_id,
                source_file = doc.source_file,
                doc_type    = doc.doc_type,
                page_number = page_num,
                chunk_index = idx,
                total_chunks= total,
            )
            all_chunks.append(chunk)

        logger.info(
            f"  '{doc.source_file}' → {total} chunks "
            f"(avg {sum(len(t) for t,_ in doc_chunks_raw)//max(total,1)} chars)"
        )

    logger.success(
        f"Chunking complete: {len(documents)} docs → {len(all_chunks)} chunks"
    )
    return all_chunks


def _clean_text(text: str) -> str:
    """
    Light cleaning for PDF-extracted text:
      - Remove ligature artifacts (ﬁ → fi, ﬂ → fl)
      - Collapse 3+ newlines to 2
      - Normalise unicode whitespace
    """
    # Common PDF ligature replacements
    replacements = {
        "\ufb01": "fi", "\ufb02": "fl", "\ufb00": "ff",
        "\ufb03": "ffi", "\ufb04": "ffl", "\u2019": "'",
        "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "--",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Normalise whitespace within lines
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


# ── Chunk stats helper ────────────────────────────────────────────────────────

def chunk_stats(chunks: list[TextChunk]) -> dict:
    """Print and return summary statistics about the chunk set."""
    if not chunks:
        return {}

    lengths = [len(c.text) for c in chunks]
    by_doc  = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, 0)
        by_doc[c.doc_id] += 1

    stats = {
        "total_chunks"   : len(chunks),
        "avg_chars"      : sum(lengths) // len(lengths),
        "min_chars"      : min(lengths),
        "max_chars"      : max(lengths),
        "docs"           : len(by_doc),
        "chunks_per_doc" : by_doc,
    }

    print(f"\n{'='*50}")
    print("CHUNK STATISTICS")
    print(f"{'='*50}")
    print(f"  Total chunks : {stats['total_chunks']}")
    print(f"  Avg length   : {stats['avg_chars']} chars")
    print(f"  Min / Max    : {stats['min_chars']} / {stats['max_chars']} chars")
    print(f"  Documents    : {stats['docs']}")
    for doc_id, count in by_doc.items():
        print(f"    {doc_id}: {count} chunks")
    print(f"{'='*50}\n")

    return stats


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.ingestion.pdf_loader import load_documents

    docs   = load_documents("data/contracts", recursive=True)
    chunks = chunk_documents(docs)
    chunk_stats(chunks)

    print("SAMPLE CHUNKS:")
    for c in chunks[:3]:
        print(f"\n  [{c.doc_id} | page {c.page_number} | chunk {c.chunk_index}]")
        print(f"  {c.text[:200]}...")
