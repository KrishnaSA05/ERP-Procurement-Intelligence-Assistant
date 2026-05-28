"""
src/vectorstore/chroma_store.py
────────────────────────────────
ChromaDB client wrapper.
Handles collection management, document upsert, and similarity search.

Two ChromaDB collections:
  • vendor_contracts      — contract PDFs (penalty, payment, termination clauses)
  • procurement_policies  — policy PDFs (approval thresholds, compliance rules)

Usage:
    from src.vectorstore.chroma_store import ChromaStore
    store = ChromaStore()
    store.upsert(embedded_chunks)
    results = store.query("late delivery penalty clause", n_results=5)
"""

import os
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings
from loguru import logger

from src.ingestion.embedder import EmbeddedChunk, embed_query


# ── Query result model ────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """A single chunk returned from a similarity search."""
    chunk_id    : str
    text        : str
    score       : float    # cosine distance (lower = more similar); converted to similarity below
    doc_id      : str
    source_file : str
    doc_type    : str
    page_number : int
    chunk_index : int

    @property
    def similarity(self) -> float:
        """Convert ChromaDB cosine distance to similarity score (0→1)."""
        return round(1 - self.score, 4)


# ── ChromaStore ───────────────────────────────────────────────────────────────

class ChromaStore:
    """
    Thin wrapper around ChromaDB with two fixed collections:
      - vendor_contracts
      - procurement_policies

    Each collection stores:
      documents  : chunk text
      embeddings : float vectors
      metadatas  : doc_id, source_file, doc_type, page_number, chunk_index
      ids        : chunk_id (stable SHA hash)
    """

    COLLECTION_CONTRACTS = "vendor_contracts"
    COLLECTION_POLICIES  = "procurement_policies"

    def __init__(
        self,
        host      : str = None,
        port      : int = None,
        persist_dir: str = None,   # only used for ephemeral local mode (no Docker)
    ):
        host = host or os.getenv("CHROMA_HOST", "localhost")
        port = int(port or os.getenv("CHROMA_PORT", 8001))

        logger.info(f"Connecting to ChromaDB at {host}:{port}")

        try:
            self._client = chromadb.HttpClient(
                host     = host,
                port     = port,
                settings = Settings(anonymized_telemetry=False),
            )
            # Verify connection
            self._client.heartbeat()
            logger.success("  ✓ ChromaDB connection established")
        except Exception as e:
            logger.warning(f"ChromaDB server unreachable ({e}). Falling back to in-memory mode.")
            self._client = chromadb.Client()

        self._contracts_col = self._get_or_create(self.COLLECTION_CONTRACTS)
        self._policies_col  = self._get_or_create(self.COLLECTION_POLICIES)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_or_create(self, name: str):
        """Get a collection if it exists, otherwise create it."""
        col = self._client.get_or_create_collection(
            name      = name,
            metadata  = {"hnsw:space": "cosine"},   # cosine similarity
        )
        count = col.count()
        logger.info(f"  Collection '{name}': {count} existing documents")
        return col

    def _collection_for(self, doc_type: str):
        """Route to the right collection based on document type."""
        if doc_type == "contract":
            return self._contracts_col
        elif doc_type == "policy":
            return self._policies_col
        else:
            # Default: contracts
            return self._contracts_col

    # ── Upsert ────────────────────────────────────────────────────────────────

    def upsert(
        self,
        embedded_chunks: list[EmbeddedChunk],
        batch_size     : int = 100,
    ) -> dict:
        """
        Insert or update chunks into ChromaDB.
        Uses upsert so re-running the pipeline is idempotent.

        Args:
            embedded_chunks : Output from embedder.embed_chunks()
            batch_size      : Chunks per ChromaDB call (avoid memory spikes)

        Returns:
            Dict with counts of upserted chunks per collection.
        """
        # Separate chunks by doc_type
        by_type: dict[str, list[EmbeddedChunk]] = {}
        for ec in embedded_chunks:
            dt = ec.chunk.doc_type
            by_type.setdefault(dt, [])
            by_type[dt].append(ec)

        counts = {}

        for doc_type, chunks in by_type.items():
            collection = self._collection_for(doc_type)
            total = len(chunks)
            n_batches = (total + batch_size - 1) // batch_size

            for b in range(n_batches):
                start = b * batch_size
                end   = min(start + batch_size, total)
                batch = chunks[start:end]

                collection.upsert(
                    ids        = [ec.chunk_id    for ec in batch],
                    documents  = [ec.text        for ec in batch],
                    embeddings = [ec.embedding   for ec in batch],
                    metadatas  = [ec.metadata    for ec in batch],
                )

            counts[collection.name] = total
            logger.success(f"  ✓ Upserted {total} chunks → '{collection.name}'")

        return counts

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(
        self,
        query_text    : str,
        n_results     : int = 5,
        doc_type      : str = None,   # "contract" | "policy" | None (search both)
        filter_doc_id : str = None,   # restrict to a specific document
        score_threshold: float = 0.3, # min similarity (0→1); lower = more permissive
    ) -> list[RetrievedChunk]:
        """
        Similarity search across one or both collections.

        Args:
            query_text      : Natural language question.
            n_results       : Max results to return.
            doc_type        : If set, only searches that collection.
            filter_doc_id   : If set, restricts to one document.
            score_threshold : Minimum similarity to include in results.

        Returns:
            List of RetrievedChunk sorted by similarity (highest first).
        """
        query_vector = embed_query(query_text)

        # Determine which collections to search
        if doc_type == "contract":
            collections = [self._contracts_col]
        elif doc_type == "policy":
            collections = [self._policies_col]
        else:
            collections = [self._contracts_col, self._policies_col]

        where_filter = {"doc_id": filter_doc_id} if filter_doc_id else None

        all_results: list[RetrievedChunk] = []

        for col in collections:
            if col.count() == 0:
                logger.debug(f"  Skipping empty collection '{col.name}'")
                continue

            try:
                result = col.query(
                    query_embeddings = [query_vector],
                    n_results        = n_results,
                    where            = where_filter,
                    include          = ["documents", "metadatas", "distances"],
                )
            except Exception as e:
                logger.warning(f"  Query failed on '{col.name}': {e}")
                continue

            ids       = result["ids"][0]
            docs      = result["documents"][0]
            metas     = result["metadatas"][0]
            distances = result["distances"][0]

            for cid, doc, meta, dist in zip(ids, docs, metas, distances):
                rc = RetrievedChunk(
                    chunk_id    = cid,
                    text        = doc,
                    score       = dist,
                    doc_id      = meta.get("doc_id", ""),
                    source_file = meta.get("source_file", ""),
                    doc_type    = meta.get("doc_type", ""),
                    page_number = meta.get("page_number", 0),
                    chunk_index = meta.get("chunk_index", 0),
                )
                if rc.similarity >= score_threshold:
                    all_results.append(rc)

        # Sort by similarity descending, return top n
        all_results.sort(key=lambda r: r.similarity, reverse=True)
        top = all_results[:n_results]

        logger.debug(
            f"Query: '{query_text[:60]}...' → {len(top)} results "
            f"(scores: {[round(r.similarity,3) for r in top]})"
        )
        return top

    # ── Utility ───────────────────────────────────────────────────────────────

    def collection_stats(self) -> dict:
        """Return row counts for each collection."""
        return {
            self.COLLECTION_CONTRACTS: self._contracts_col.count(),
            self.COLLECTION_POLICIES : self._policies_col.count(),
        }

    def reset_collection(self, doc_type: str):
        """Delete and recreate a collection. Useful for re-ingestion."""
        name = (
            self.COLLECTION_CONTRACTS if doc_type == "contract"
            else self.COLLECTION_POLICIES
        )
        self._client.delete_collection(name)
        logger.warning(f"Deleted collection '{name}'")
        if doc_type == "contract":
            self._contracts_col = self._get_or_create(self.COLLECTION_CONTRACTS)
        else:
            self._policies_col = self._get_or_create(self.COLLECTION_POLICIES)

    def reset_all(self):
        """Wipe both collections. WARNING: destructive."""
        self.reset_collection("contract")
        self.reset_collection("policy")
        logger.warning("Both collections reset.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    store = ChromaStore()
    print("\nCollection stats:", store.collection_stats())

    test_query = "late delivery penalty clause"
    print(f"\nTest query: '{test_query}'")
    results = store.query(test_query, n_results=3)

    if not results:
        print("  No results (collections may be empty — run ingest_pipeline.py first)")
    else:
        for r in results:
            print(f"\n  [{r.doc_id} | p.{r.page_number} | sim={r.similarity}]")
            print(f"  {r.text[:200]}...")
