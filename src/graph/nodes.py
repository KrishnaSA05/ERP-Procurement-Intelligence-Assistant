"""
src/graph/nodes.py
───────────────────
LangGraph node functions — one per processing step.

Each node:
  • Receives the current ProcurementState
  • Does its work (classify / query / retrieve / synthesise)
  • Returns a dict of state fields to update

Nodes are stateless functions — all shared objects (agents, stores)
are injected at graph-build time via closures to keep nodes pure.

Node map:
  classify_node   → reads: question       writes: route, route_confidence, route_reasoning
  sql_node        → reads: question       writes: sql_result
  rag_node        → reads: question       writes: rag_result
  synthesis_node  → reads: question, route, sql_result, rag_result
                    writes: final_response
"""

from loguru import logger

from src.graph.state            import ProcurementState
from src.agents.query_classifier import route_query
from src.agents.sql_agent       import SQLAgent
from src.agents.rag_agent       import RAGAgent
from src.agents.synthesis_agent import SynthesisAgent


# ── Node: classify ────────────────────────────────────────────────────────────

def classify_node(state: ProcurementState) -> dict:
    """
    Node 1 — Query Classifier.
    Determines whether to route to SQL, RAG, or both.
    """
    question = state["question"]
    logger.info(f"[NODE: classify] '{question[:70]}'")

    decision = route_query(question, use_fast_path=True)

    return {
        "route"           : decision.route.value,
        "route_confidence": decision.confidence,
        "route_reasoning" : decision.reasoning,
        "errors"          : state.get("errors", []),
    }


# ── Node factories (inject agents as dependencies) ────────────────────────────

def make_sql_node(sql_agent: SQLAgent):
    """
    Factory that returns a sql_node closure with the agent injected.
    Avoids re-initialising the agent on every graph invocation.
    """
    def sql_node(state: ProcurementState) -> dict:
        """
        Node 2a — SQL Agent.
        Runs only for 'sql' and 'hybrid' routes.
        """
        question = state["question"]
        logger.info(f"[NODE: sql] '{question[:70]}'")

        errors = list(state.get("errors", []))

        try:
            sql_result = sql_agent.run(question)
            if not sql_result.success:
                errors.append(f"SQL agent error: {sql_result.error}")
        except Exception as e:
            logger.error(f"  SQL node exception: {e}")
            from src.agents.sql_agent import SQLAgentResult
            sql_result = SQLAgentResult(
                question=question, sql_query="", data=[],
                narrative="SQL query failed.", success=False, error=str(e)
            )
            errors.append(f"SQL node exception: {e}")

        return {"sql_result": sql_result, "errors": errors}

    return sql_node


def make_rag_node(rag_agent: RAGAgent):
    """
    Factory that returns a rag_node closure with the agent injected.
    """
    def rag_node(state: ProcurementState) -> dict:
        """
        Node 2b — RAG Agent.
        Runs only for 'rag' and 'hybrid' routes.
        """
        question = state["question"]
        logger.info(f"[NODE: rag] '{question[:70]}'")

        errors = list(state.get("errors", []))

        try:
            rag_result = rag_agent.run(question)
            if not rag_result.success:
                errors.append(f"RAG agent error: {rag_result.error}")
        except Exception as e:
            logger.error(f"  RAG node exception: {e}")
            from src.agents.rag_agent import RAGAgentResult
            rag_result = RAGAgentResult(
                question=question, answer="Document retrieval failed.",
                success=False, error=str(e)
            )
            errors.append(f"RAG node exception: {e}")

        return {"rag_result": rag_result, "errors": errors}

    return rag_node


def make_synthesis_node(synthesis_agent: SynthesisAgent):
    """
    Factory that returns a synthesis_node closure with the agent injected.
    """
    def synthesis_node(state: ProcurementState) -> dict:
        """
        Node 3 — Synthesis Agent.
        Always runs last — merges whatever results exist.
        """
        question   = state["question"]
        route      = state.get("route", "hybrid")
        sql_result = state.get("sql_result")
        rag_result = state.get("rag_result")

        logger.info(f"[NODE: synthesis] route={route.upper()}")

        errors = list(state.get("errors", []))

        try:
            final_response = synthesis_agent.synthesise(
                question   = question,
                route      = route,
                sql_result = sql_result,
                rag_result = rag_result,
            )
        except Exception as e:
            logger.error(f"  Synthesis node exception: {e}")
            from src.agents.synthesis_agent import FinalResponse
            final_response = FinalResponse(
                question     = question,
                final_answer = f"Synthesis failed: {e}",
                route_used   = route,
                success      = False,
                error        = str(e),
            )
            errors.append(f"Synthesis exception: {e}")

        return {"final_response": final_response, "errors": errors}

    return synthesis_node


# ── Routing function (used in conditional edges) ──────────────────────────────

def route_after_classify(state: ProcurementState) -> str:
    """
    Conditional edge function called after classify_node.
    Returns the next node name based on the route decision.

    LangGraph uses this return value to decide which node to visit next.
    """
    route = state.get("route", "hybrid")
    logger.debug(f"  Routing to: {route}")

    if route == "sql":
        return "sql_node"
    elif route == "rag":
        return "rag_node"
    else:
        return "sql_node"    # hybrid: go to SQL first, then RAG


def route_after_sql(state: ProcurementState) -> str:
    """
    Conditional edge after sql_node.
    For hybrid: go to RAG next.
    For sql-only: go straight to synthesis.
    """
    route = state.get("route", "sql")
    if route == "hybrid":
        return "rag_node"
    return "synthesis_node"
