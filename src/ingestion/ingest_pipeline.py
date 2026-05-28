"""
src/ingestion/ingest_pipeline.py
──────────────────────────────────
Master ingestion pipeline. Run this once after adding new PDF documents.

Pipeline:
  1. Load PDFs     (pdf_loader.py)
  2. Chunk text    (chunker.py)
  3. Embed chunks  (embedder.py)
  4. Upsert to ChromaDB  (chroma_store.py)

Usage:
    # Ingest contracts only
    python src/ingestion/ingest_pipeline.py --folder data/contracts

    # Ingest policies only
    python src/ingestion/ingest_pipeline.py --folder data/policies

    # Ingest everything (contracts + policies)
    python src/ingestion/ingest_pipeline.py --all

    # Reset and re-ingest
    python src/ingestion/ingest_pipeline.py --all --reset
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from loguru import logger

from src.ingestion.pdf_loader  import load_documents
from src.ingestion.chunker     import chunk_documents, chunk_stats
from src.ingestion.embedder    import get_embedder, embed_chunks
from src.vectorstore.chroma_store import ChromaStore

load_dotenv()


def run_pipeline(
    folder   : str | Path,
    embedder = None,
    store    : ChromaStore = None,
    reset    : bool = False,
) -> dict:
    """
    Run the full ingestion pipeline for one folder.

    Args:
        folder   : Path to folder containing PDFs.
        embedder : Embedder instance (shared across calls to avoid reloading model).
        store    : ChromaStore instance (shared).
        reset    : If True, wipe and re-ingest the relevant collection.

    Returns:
        Summary dict with counts.
    """
    folder = Path(folder)
    logger.info(f"\n{'='*55}")
    logger.info(f"INGESTING: {folder}")
    logger.info(f"{'='*55}")

    t0 = time.time()

    # ── Step 1: Load ──────────────────────────────────────────────────────────
    logger.info("Step 1/4 — Loading PDFs...")
    documents = load_documents(folder, recursive=True)
    if not documents:
        logger.warning(f"No PDFs found in {folder}. Skipping.")
        return {"folder": str(folder), "docs": 0, "chunks": 0, "status": "skipped"}

    # ── Step 2: Chunk ─────────────────────────────────────────────────────────
    logger.info("Step 2/4 — Chunking...")
    chunks = chunk_documents(documents, chunk_size=800, overlap=150)
    chunk_stats(chunks)

    # ── Step 3: Embed ─────────────────────────────────────────────────────────
    logger.info("Step 3/4 — Embedding...")
    if embedder is None:
        embedder = get_embedder()
    embedded = embed_chunks(chunks, embedder)

    # ── Step 4: Upsert ───────────────────────────────────────────────────────
    logger.info("Step 4/4 — Upserting to ChromaDB...")
    if store is None:
        store = ChromaStore()

    # Optionally reset before ingesting
    if reset:
        doc_type = documents[0].doc_type if documents else "contract"
        logger.warning(f"  --reset: wiping '{doc_type}' collection before ingestion")
        store.reset_collection(doc_type)

    counts = store.upsert(embedded)

    elapsed = round(time.time() - t0, 1)
    result = {
        "folder"  : str(folder),
        "docs"    : len(documents),
        "chunks"  : len(chunks),
        "embedded": len(embedded),
        "upserted": counts,
        "elapsed" : f"{elapsed}s",
        "status"  : "success",
    }

    logger.success(
        f"Pipeline complete: {len(documents)} docs → "
        f"{len(chunks)} chunks → {len(embedded)} embeddings "
        f"({elapsed}s)"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="ERP Procurement Document Ingestion Pipeline")
    parser.add_argument("--folder", type=str, help="Path to a specific folder of PDFs")
    parser.add_argument("--all",    action="store_true", help="Ingest both contracts and policies")
    parser.add_argument("--reset",  action="store_true", help="Reset collection before ingesting")
    args = parser.parse_args()

    if not args.folder and not args.all:
        parser.print_help()
        sys.exit(1)

    # Shared instances — avoids reloading the embedding model multiple times
    embedder = get_embedder()
    store    = ChromaStore()

    results = []

    if args.all:
        for folder in ["data/contracts", "data/policies"]:
            p = Path(folder)
            if p.exists():
                result = run_pipeline(p, embedder=embedder, store=store, reset=args.reset)
                results.append(result)
            else:
                logger.warning(f"Folder not found: {folder} — skipping")
    else:
        result = run_pipeline(args.folder, embedder=embedder, store=store, reset=args.reset)
        results.append(result)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("INGESTION SUMMARY")
    print(f"{'='*55}")
    for r in results:
        print(f"  {r['folder']}")
        print(f"    docs    : {r['docs']}")
        print(f"    chunks  : {r['chunks']}")
        print(f"    status  : {r['status']}")
        print(f"    time    : {r.get('elapsed', 'n/a')}")
    print(f"\nCOLLECTION STATS:")
    stats = store.collection_stats()
    for name, count in stats.items():
        print(f"  {name}: {count} chunks")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
