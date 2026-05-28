"""
src/agents/synthesis_agent.py
───────────────────────────────
Agent 3 — Synthesis Agent.

Takes outputs from the SQL Agent and/or RAG Agent and merges them
into a single, coherent, cited response for the user.

Handles three cases:
  • SQL only   — wraps the narrative cleanly
  • RAG only   — wraps the answer + citations cleanly
  • HYBRID     — genuinely merges both, cross-referencing findings

Usage:
    from src.agents.synthesis_agent import SynthesisAgent
    from src.agents.sql_agent import SQLAgentResult
    from src.agents.rag_agent import RAGAgentResult

    agent  = SynthesisAgent()
    result = agent.synthesise(
        question   = "Which vendors have open POs above 50K and no penalty clause?",
        sql_result = sql_result,
        rag_result = rag_result,
    )
    print(result.final_answer)
"""

from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from src.agents.bedrock_client  import get_llm
from src.agents.sql_agent       import SQLAgentResult
from src.agents.rag_agent       import RAGAgentResult, Citation


# ── Final response model ──────────────────────────────────────────────────────

@dataclass
class FinalResponse:
    question      : str
    final_answer  : str
    route_used    : str                      # "sql" | "rag" | "hybrid"
    sql_query     : str = ""                 # shown in UI for transparency
    citations     : list[Citation] = field(default_factory=list)
    data_rows     : list[dict]     = field(default_factory=list)
    success       : bool = True
    error         : str  = ""

    def to_dict(self) -> dict:
        return {
            "question"    : self.question,
            "final_answer": self.final_answer,
            "route_used"  : self.route_used,
            "sql_query"   : self.sql_query,
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
            "data_rows"   : self.data_rows,
            "success"     : self.success,
            "error"       : self.error,
        }


# ── Prompts ───────────────────────────────────────────────────────────────────

HYBRID_SYNTHESIS_SYSTEM = """You are a senior procurement analyst writing a combined intelligence briefing.

You have been given two pieces of analysis:
  1. STRUCTURED DATA — results from querying the ERP database (purchase orders, vendors, invoices)
  2. DOCUMENT INTELLIGENCE — findings from vendor contracts and procurement policy documents

Your job is to synthesise both into a single, clear, actionable response.

RULES:
1. Start with the direct answer to the question in 1-2 sentences.
2. Present the data findings first (numbers, vendor names, amounts).
3. Then present the document findings (clauses, policy rules).
4. Explicitly connect the two — show how the document findings apply to the data findings.
5. End with a clear business implication or recommendation if appropriate.
6. Keep the total response to 150-250 words.
7. Use business language — no SQL, no technical jargon.
8. Format currency as $X,XXX. Use bullet points only when listing 3+ items."""

HYBRID_SYNTHESIS_HUMAN = """Question: {question}

─── ERP DATA FINDINGS ───────────────────────────────────
{sql_narrative}

Raw data summary ({row_count} records):
{data_preview}

─── DOCUMENT FINDINGS ───────────────────────────────────
{rag_answer}

Sources: {sources}

─────────────────────────────────────────────────────────
Write a unified procurement intelligence briefing that combines both findings:"""


# ── Synthesis Agent ───────────────────────────────────────────────────────────

class SynthesisAgent:
    """
    Merges SQL and RAG agent outputs into a single response.

    For SQL-only or RAG-only routes, it reformats cleanly.
    For HYBRID routes, it genuinely merges and cross-references both.
    """

    def __init__(self):
        self._llm = get_llm(temperature=0.3, max_tokens=1024)
        logger.info("SynthesisAgent initialised")

    # ── SQL only ──────────────────────────────────────────────────────────────

    def _wrap_sql(
        self,
        question   : str,
        sql_result : SQLAgentResult,
    ) -> FinalResponse:
        """For SQL-only routes — the narrative is already good, just package it."""
        if not sql_result.success:
            return FinalResponse(
                question     = question,
                final_answer = f"I was unable to retrieve data from the ERP system. {sql_result.error}",
                route_used   = "sql",
                success      = False,
                error        = sql_result.error,
            )

        return FinalResponse(
            question     = question,
            final_answer = sql_result.narrative,
            route_used   = "sql",
            sql_query    = sql_result.sql_query,
            data_rows    = sql_result.data,
            success      = True,
        )

    # ── RAG only ──────────────────────────────────────────────────────────────

    def _wrap_rag(
        self,
        question   : str,
        rag_result : RAGAgentResult,
    ) -> FinalResponse:
        """For RAG-only routes — package the answer with citations."""
        if not rag_result.success:
            return FinalResponse(
                question     = question,
                final_answer = f"I was unable to retrieve information from the documents. {rag_result.error}",
                route_used   = "rag",
                success      = False,
                error        = rag_result.error,
            )

        return FinalResponse(
            question     = question,
            final_answer = rag_result.answer,
            route_used   = "rag",
            citations    = rag_result.citations,
            success      = True,
        )

    # ── Hybrid ────────────────────────────────────────────────────────────────

    def _merge_hybrid(
        self,
        question   : str,
        sql_result : SQLAgentResult,
        rag_result : RAGAgentResult,
    ) -> FinalResponse:
        """For HYBRID routes — LLM synthesises both agent outputs together."""

        # If one agent failed, degrade gracefully
        if not sql_result.success and not rag_result.success:
            return FinalResponse(
                question     = question,
                final_answer = "Both data and document retrieval failed. Please check system connectivity.",
                route_used   = "hybrid",
                success      = False,
                error        = f"SQL: {sql_result.error} | RAG: {rag_result.error}",
            )

        if not sql_result.success:
            logger.warning("  SQL agent failed in hybrid — falling back to RAG only")
            result = self._wrap_rag(question, rag_result)
            result.route_used = "hybrid"
            return result

        if not rag_result.success:
            logger.warning("  RAG agent failed in hybrid — falling back to SQL only")
            result = self._wrap_sql(question, sql_result)
            result.route_used = "hybrid"
            return result

        # Both succeeded — synthesise properly
        data_preview = self._format_data_preview(sql_result.data)
        sources      = ", ".join(
            c.source_file for c in rag_result.citations[:3]
        ) or "No documents retrieved"

        messages = [
            SystemMessage(content=HYBRID_SYNTHESIS_SYSTEM),
            HumanMessage(content=HYBRID_SYNTHESIS_HUMAN.format(
                question      = question,
                sql_narrative = sql_result.narrative,
                row_count     = sql_result.row_count,
                data_preview  = data_preview,
                rag_answer    = rag_result.answer,
                sources       = sources,
            )),
        ]

        response     = self._llm.invoke(messages)
        final_answer = response.content.strip()

        return FinalResponse(
            question     = question,
            final_answer = final_answer,
            route_used   = "hybrid",
            sql_query    = sql_result.sql_query,
            citations    = rag_result.citations,
            data_rows    = sql_result.data,
            success      = True,
        )

    def _format_data_preview(self, data: list[dict], max_rows: int = 5) -> str:
        """Format raw data rows into a readable text preview."""
        if not data:
            return "No records found."
        preview = data[:max_rows]
        lines   = []
        for row in preview:
            parts = [f"{k}: {v}" for k, v in row.items()]
            lines.append("  • " + " | ".join(parts))
        if len(data) > max_rows:
            lines.append(f"  ... and {len(data) - max_rows} more rows")
        return "\n".join(lines)

    # ── Main entry point ──────────────────────────────────────────────────────

    def synthesise(
        self,
        question   : str,
        route      : str,
        sql_result : SQLAgentResult | None = None,
        rag_result : RAGAgentResult | None = None,
    ) -> FinalResponse:
        """
        Synthesise agent outputs into a final response.

        Args:
            question   : Original user question.
            route      : "sql" | "rag" | "hybrid"
            sql_result : Output from SQLAgent.run() — required for sql/hybrid routes.
            rag_result : Output from RAGAgent.run() — required for rag/hybrid routes.

        Returns:
            FinalResponse with final_answer, citations, and data.
        """
        logger.info(f"SynthesisAgent: route={route.upper()}")

        if route == "sql":
            if sql_result is None:
                return FinalResponse(question=question,
                    final_answer="SQL result missing.", route_used="sql",
                    success=False, error="sql_result is None")
            return self._wrap_sql(question, sql_result)

        elif route == "rag":
            if rag_result is None:
                return FinalResponse(question=question,
                    final_answer="RAG result missing.", route_used="rag",
                    success=False, error="rag_result is None")
            return self._wrap_rag(question, rag_result)

        elif route == "hybrid":
            if sql_result is None or rag_result is None:
                # Degrade to whichever is available
                if sql_result:
                    return self._wrap_sql(question, sql_result)
                if rag_result:
                    return self._wrap_rag(question, rag_result)
                return FinalResponse(question=question,
                    final_answer="No results available.", route_used="hybrid",
                    success=False, error="Both results are None")
            return self._merge_hybrid(question, sql_result, rag_result)

        else:
            logger.warning(f"  Unknown route '{route}' — defaulting to hybrid")
            return self.synthesise(question, "hybrid", sql_result, rag_result)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.agents.sql_agent import SQLAgent
    from src.agents.rag_agent import RAGAgent

    sql_agent   = SQLAgent()
    rag_agent   = RAGAgent()
    synth_agent = SynthesisAgent()

    # Test HYBRID query
    question = (
        "Which vendors have open POs above $50,000 "
        "and what do their contracts say about late delivery penalties?"
    )

    print(f"\n{'='*60}")
    print(f"Q: {question}")
    print(f"{'='*60}")

    sql_result = sql_agent.run(
        "Which vendors have open purchase orders above $50,000?"
    )
    rag_result = rag_agent.run(
        "What are the late delivery penalty clauses in vendor contracts?"
    )

    final = synth_agent.synthesise(
        question   = question,
        route      = "hybrid",
        sql_result = sql_result,
        rag_result = rag_result,
    )

    print(f"\nFINAL ANSWER:\n{final.final_answer}")
    print(f"\nROUTE USED: {final.route_used.upper()}")
    print(f"\nSQL QUERY:\n{final.sql_query}")
    print(f"\nCITATIONS:")
    for c in final.citations:
        print(f"  {c}")
