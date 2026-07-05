"""
src/agents/rag_agent.py
────────────────────────
Agent 2b — RAG Agent.

Retrieves relevant contract/policy chunks from ChromaDB and
generates a cited answer using Claude Haiku.

Pipeline:
  1. Embed the query
  2. Similarity search across ChromaDB collections
  3. Re-rank retrieved chunks by relevance score
  4. Build context window from top-k chunks
  5. Claude Haiku generates answer grounded in the context
  6. Returns RAGAgentResult with answer + source citations

Usage:
    from src.agents.rag_agent import RAGAgent
    agent  = RAGAgent()
    result = agent.run("Does our contract include a late delivery penalty clause?")
    print(result.answer)
    for c in result.citations:
        print(c)
"""

import json
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from src.agents.bedrock_client import get_llm
from src.vectorstore.chroma_store import ChromaStore, RetrievedChunk


# ── Result model ──────────────────────────────────────────────────────────────

@dataclass
class Citation:
    source_file : str
    doc_id      : str
    doc_type    : str
    page_number : int
    similarity  : float
    excerpt     : str       # first 200 chars of the chunk

    def __str__(self) -> str:
        return (
            f"[{self.doc_type.upper()}] {self.source_file} "
            f"(page {self.page_number}, similarity={self.similarity:.3f})"
        )


@dataclass
class RAGAgentResult:
    question    : str
    answer      : str
    citations   : list[Citation] = field(default_factory=list)
    chunks_used : int = 0
    success     : bool = True
    error       : str = ""

    def citations_text(self) -> str:
        """Formatted citation list for display."""
        if not self.citations:
            return "No source documents found."
        lines = ["Sources:"]
        for i, c in enumerate(self.citations, 1):
            lines.append(f"  [{i}] {c}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "question"    : self.question,
            "answer"      : self.answer,
            "citations"   : [
                {
                    "source_file": c.source_file,
                    "doc_type"   : c.doc_type,
                    "page_number": c.page_number,
                    "similarity" : c.similarity,
                    "excerpt"    : c.excerpt,
                }
                for c in self.citations
            ],
            "chunks_used" : self.chunks_used,
            "success"     : self.success,
            "error"       : self.error,
        }


# ── Prompts ───────────────────────────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """You are a procurement intelligence assistant with access to vendor contracts and procurement policy documents.

Your job is to answer questions using ONLY the document excerpts provided as context.

RULES:
1. Base your answer EXCLUSIVELY on the provided context — do not use general knowledge.
2. If the context contains a direct answer, quote the relevant clause or policy language.
3. If the context does not contain enough information to answer, say so clearly.
4. Always indicate which document your answer comes from (e.g. "According to the Alpha Tech contract..." or "The procurement policy states...").
5. Be specific — reference clause numbers, page numbers, or section titles when visible.
6. Keep answers concise: 3-6 sentences for simple questions, up to 10 for complex ones.
7. Never fabricate clauses or policy rules that are not in the provided context."""

RAG_HUMAN_TEMPLATE = """Context from procurement documents:
─────────────────────────────────────
{context}
─────────────────────────────────────

Question: {question}

Answer based strictly on the context above:"""


# ── Re-ranker ─────────────────────────────────────────────────────────────────

def rerank_chunks(
    chunks   : list[RetrievedChunk],
    query    : str,
    top_k    : int = 5,
) -> list[RetrievedChunk]:
    """
    Re-rank retrieved chunks using a simple cross-encoder heuristic.

    Strategy: boost chunks that contain query keywords,
    then sort by (keyword_boost * 0.3 + similarity * 0.7).

    Note: In production you'd use a proper cross-encoder
    (e.g. cross-encoder/ms-marco-MiniLM-L-6-v2), but this
    keyword-boosted rerank works well for procurement queries
    and avoids an extra model dependency.
    """
    query_words = set(query.lower().split())

    def score(chunk: RetrievedChunk) -> float:
        chunk_words    = set(chunk.text.lower().split())
        overlap        = len(query_words & chunk_words) / max(len(query_words), 1)
        keyword_boost  = min(overlap * 2, 1.0)   # cap at 1.0
        return 0.7 * chunk.similarity + 0.3 * keyword_boost

    reranked = sorted(chunks, key=score, reverse=True)
    return reranked[:top_k]


# ── Context builder ───────────────────────────────────────────────────────────

def build_context(chunks: list[RetrievedChunk], max_chars: int = 6000) -> str:
    """
    Build a context string from retrieved chunks.
    Caps total length to stay within Claude Haiku's context window.
    Each chunk is labelled with its source for attribution.
    """
    context_parts = []
    total_chars   = 0

    for i, chunk in enumerate(chunks, 1):
        label   = (
            f"[Excerpt {i} | {chunk.source_file} | "
            f"Page {chunk.page_number} | Similarity: {chunk.similarity:.3f}]"
        )
        section = f"{label}\n{chunk.text}"

        if total_chars + len(section) > max_chars:
            # Truncate last chunk to fit
            remaining = max_chars - total_chars
            if remaining > 200:
                section = section[:remaining] + "..."
                context_parts.append(section)
            break

        context_parts.append(section)
        total_chars += len(section)

    return "\n\n".join(context_parts)


# ── RAG Agent ─────────────────────────────────────────────────────────────────

class RAGAgent:
    """
    Retrieval-Augmented Generation agent for procurement documents.

    Retrieves relevant chunks from ChromaDB, re-ranks them,
    and generates a cited answer using Claude Haiku.
    """

    def __init__(
        self,
        store      : ChromaStore = None,
        n_retrieve : int = 10,    # initial retrieval count (before re-ranking)
        n_final    : int = 5,     # final chunks passed to LLM after re-ranking
    ):
        self._store      = store or ChromaStore()
        self._llm        = get_llm(temperature=0.1, max_tokens=1024, label="rag_agent")
        self._n_retrieve = n_retrieve
        self._n_final    = n_final
        logger.info(f"RAGAgent initialised (retrieve={n_retrieve}, final={n_final})")

    # ── Step 1 + 2: Retrieve and re-rank ─────────────────────────────────────

    def _retrieve(
        self,
        query    : str,
        doc_type : str = None,
    ) -> list[RetrievedChunk]:
        """Retrieve and re-rank chunks from ChromaDB."""
        raw_chunks = self._store.query(
            query_text     = query,
            n_results      = self._n_retrieve,
            doc_type       = doc_type,
            score_threshold= 0.25,
        )

        if not raw_chunks:
            logger.warning(f"  No chunks retrieved for: '{query[:60]}'")
            return []

        reranked = rerank_chunks(raw_chunks, query, top_k=self._n_final)
        logger.debug(
            f"  Retrieved {len(raw_chunks)} → re-ranked to {len(reranked)} chunks"
        )
        return reranked

    # ── Step 3: Generate answer ───────────────────────────────────────────────

    def _generate_answer(
        self,
        question : str,
        chunks   : list[RetrievedChunk],
    ) -> str:
        """Generate a grounded answer from context chunks."""
        if not chunks:
            return (
                "I could not find relevant information in the vendor contracts or "
                "procurement policy documents to answer this question."
            )

        context = build_context(chunks)

        messages = [
            SystemMessage(content=RAG_SYSTEM_PROMPT),
            HumanMessage(content=RAG_HUMAN_TEMPLATE.format(
                context  = context,
                question = question,
            )),
        ]

        response = self._llm.invoke(messages)
        return response.content.strip()

    # ── Main run ──────────────────────────────────────────────────────────────

    def run(
        self,
        question : str,
        doc_type : str = None,     # "contract" | "policy" | None (search both)
    ) -> RAGAgentResult:
        """
        Full RAG pipeline: question → retrieve → re-rank → generate → cite.

        Args:
            question : Natural language procurement question.
            doc_type : Restrict to a specific collection, or None to search both.

        Returns:
            RAGAgentResult with answer and citations.
        """
        logger.info(f"RAGAgent running: '{question[:80]}'")

        try:
            # Retrieve
            chunks = self._retrieve(question, doc_type=doc_type)

            # Generate
            answer = self._generate_answer(question, chunks)

            # Build citations
            citations = [
                Citation(
                    source_file = c.source_file,
                    doc_id      = c.doc_id,
                    doc_type    = c.doc_type,
                    page_number = c.page_number,
                    similarity  = c.similarity,
                    excerpt     = c.text[:200].replace("\n", " "),
                )
                for c in chunks
            ]

            # Deduplicate citations by source_file (keep highest similarity)
            seen = {}
            for cit in citations:
                if cit.source_file not in seen or cit.similarity > seen[cit.source_file].similarity:
                    seen[cit.source_file] = cit
            unique_citations = sorted(seen.values(), key=lambda x: x.similarity, reverse=True)

            result = RAGAgentResult(
                question    = question,
                answer      = answer,
                citations   = unique_citations,
                chunks_used = len(chunks),
                success     = True,
            )

            logger.success(
                f"  RAGAgent done: {len(chunks)} chunks → "
                f"{len(unique_citations)} unique sources cited"
            )
            return result

        except Exception as e:
            logger.error(f"  RAGAgent error: {e}")
            return RAGAgentResult(
                question = question,
                answer   = f"An error occurred during document retrieval: {e}",
                success  = False,
                error    = str(e),
            )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    agent = RAGAgent()

    test_questions = [
        "Does our vendor contract include a late delivery penalty clause?",
        "What are the payment terms in the standard vendor contract?",
        "What is our policy for single-source procurement above $100,000?",
        "What happens in case of force majeure under the contract?",
    ]

    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        result = agent.run(q)
        print(f"\nAnswer:\n{result.answer}")
        print(f"\n{result.citations_text()}")
        print(f"\nChunks used: {result.chunks_used}")
