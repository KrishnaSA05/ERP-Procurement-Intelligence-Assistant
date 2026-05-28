"""
src/ingestion/embedder.py
──────────────────────────
Generates vector embeddings for TextChunks.

Two backends — auto-selected by APP_ENV:
  • Local (development) : sentence-transformers all-MiniLM-L6-v2
                          384-dim, fast, no API cost, good quality
  • AWS (production)    : Amazon Bedrock Titan Embed Text v2
                          1536-dim, production-grade

Usage:
    from src.ingestion.embedder import get_embedder, embed_chunks
    embedder = get_embedder()
    chunks_with_vectors = embed_chunks(chunks, embedder)
"""

import os
import time
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass

import boto3
import numpy as np
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.ingestion.chunker import TextChunk


# ── Embedded chunk (chunk + its vector) ──────────────────────────────────────

@dataclass
class EmbeddedChunk:
    chunk     : TextChunk
    embedding : list[float]   # raw float vector

    @property
    def chunk_id(self)    -> str: return self.chunk.chunk_id
    @property
    def text(self)        -> str: return self.chunk.text
    @property
    def metadata(self)    -> dict:
        return {
            "doc_id"      : self.chunk.doc_id,
            "source_file" : self.chunk.source_file,
            "doc_type"    : self.chunk.doc_type,
            "page_number" : self.chunk.page_number,
            "chunk_index" : self.chunk.chunk_index,
            "total_chunks": self.chunk.total_chunks,
        }


# ── Abstract base ─────────────────────────────────────────────────────────────

class BaseEmbedder(ABC):
    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of strings. Returns list of float vectors."""
        pass

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        return self.embed_texts([query])[0]

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass


# ── Local embedder (sentence-transformers) ─────────────────────────────────────

class LocalEmbedder(BaseEmbedder):
    """
    Uses sentence-transformers all-MiniLM-L6-v2.
    384 dimensions, ~80MB model, no API cost.
    Perfect for local development and portfolio demos.
    """

    MODEL = "all-MiniLM-L6-v2"

    def __init__(self):
        logger.info(f"Loading local embedding model: {self.MODEL}")
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.MODEL)
        logger.success(f"  ✓ Local embedder ready ({self.dimension}d)")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,    # cosine similarity friendly
        )
        return vectors.tolist()

    @property
    def dimension(self) -> int:
        return 384

    @property
    def model_name(self) -> str:
        return self.MODEL


# ── Bedrock Titan embedder ────────────────────────────────────────────────────

class BedrockEmbedder(BaseEmbedder):
    """
    Uses Amazon Bedrock Titan Embed Text v2.
    1536 dimensions, production-grade, ~$0.00002 per 1K tokens.
    Used in AWS deployment.
    """

    MODEL_ID = "amazon.titan-embed-text-v2:0"

    def __init__(self, region: str = None):
        region = region or os.getenv("AWS_REGION", "us-east-1")
        logger.info(f"Initialising Bedrock embedder [{self.MODEL_ID}] in {region}")
        self._client = boto3.client("bedrock-runtime", region_name=region)
        logger.success("  ✓ Bedrock embedder ready (1536d)")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _embed_single(self, text: str) -> list[float]:
        body = json.dumps({"inputText": text[:8000]})   # Titan token limit guard
        response = self._client.invoke_model(
            modelId     = self.MODEL_ID,
            body        = body,
            contentType = "application/json",
            accept      = "application/json",
        )
        result = json.loads(response["body"].read())
        return result["embedding"]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Bedrock doesn't support batching — embed one at a time with rate control."""
        embeddings = []
        for i, text in enumerate(texts):
            emb = self._embed_single(text)
            embeddings.append(emb)
            # Polite rate limiting
            if (i + 1) % 10 == 0:
                logger.debug(f"  Embedded {i+1}/{len(texts)}")
                time.sleep(0.1)
        return embeddings

    @property
    def dimension(self) -> int:
        return 1536

    @property
    def model_name(self) -> str:
        return self.MODEL_ID


# ── Factory ───────────────────────────────────────────────────────────────────

def get_embedder() -> BaseEmbedder:
    """
    Returns the appropriate embedder based on APP_ENV.
      development → LocalEmbedder (sentence-transformers)
      production  → BedrockEmbedder (Titan Embed v2)
    """
    env = os.getenv("APP_ENV", "development").lower()
    if env == "production":
        logger.info("APP_ENV=production → using Bedrock Titan embedder")
        return BedrockEmbedder()
    else:
        logger.info("APP_ENV=development → using local sentence-transformers embedder")
        return LocalEmbedder()


# ── Main embedding function ───────────────────────────────────────────────────

def embed_chunks(
    chunks   : list[TextChunk],
    embedder : BaseEmbedder = None,
    batch_size: int = 64,
) -> list[EmbeddedChunk]:
    """
    Embed a list of TextChunks.

    Args:
        chunks     : Output from chunker.chunk_documents()
        embedder   : Embedder instance. If None, auto-selects via get_embedder().
        batch_size : How many chunks to embed per call (for local embedder).

    Returns:
        List of EmbeddedChunk objects ready for ChromaDB insertion.
    """
    if embedder is None:
        embedder = get_embedder()

    logger.info(
        f"Embedding {len(chunks)} chunks with {embedder.model_name} "
        f"({embedder.dimension}d)..."
    )

    embedded = []
    total_batches = (len(chunks) + batch_size - 1) // batch_size

    for batch_num in range(total_batches):
        start = batch_num * batch_size
        end   = min(start + batch_size, len(chunks))
        batch = chunks[start:end]

        texts   = [c.text for c in batch]
        vectors = embedder.embed_texts(texts)

        for chunk, vector in zip(batch, vectors):
            embedded.append(EmbeddedChunk(chunk=chunk, embedding=vector))

        logger.debug(f"  Batch {batch_num+1}/{total_batches} embedded ({end} / {len(chunks)})")

    logger.success(f"  ✓ {len(embedded)} chunks embedded ({embedder.dimension}d vectors)")
    return embedded


# ── Query embedding helper ────────────────────────────────────────────────────

_embedder_cache: BaseEmbedder | None = None

def embed_query(query: str) -> list[float]:
    """
    Embed a single query string. Caches the embedder instance.
    Used by the RAG agent at query time.
    """
    global _embedder_cache
    if _embedder_cache is None:
        _embedder_cache = get_embedder()
    return _embedder_cache.embed_query(query)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.ingestion.pdf_loader import load_documents
    from src.ingestion.chunker   import chunk_documents

    docs    = load_documents("data/contracts", recursive=True)
    chunks  = chunk_documents(docs)
    emb     = get_embedder()
    results = embed_chunks(chunks[:5], emb)   # test with 5 chunks

    print(f"\nSample embedding for chunk '{results[0].chunk_id}':")
    print(f"  Dims   : {len(results[0].embedding)}")
    print(f"  First 5: {results[0].embedding[:5]}")
    print(f"  Text   : {results[0].text[:120]}...")
