"""
src/graph/state.py
───────────────────
Defines the shared state object that flows through every node
in the LangGraph procurement workflow.

LangGraph passes this state between nodes — each node reads
what it needs and writes its output back into the state.

State fields progress through the graph:
  START
    │  question populated
    ▼
  classify_node
    │  route, route_confidence, route_reasoning populated
    ▼
  sql_node / rag_node / both (depending on route)
    │  sql_result and/or rag_result populated
    ▼
  synthesis_node
    │  final_response populated
    ▼
  END
"""

from typing import Optional, Any
from typing_extensions import TypedDict


class ProcurementState(TypedDict, total=False):
    """
    Shared state for the procurement intelligence LangGraph workflow.

    Fields are optional (total=False) so nodes can add them incrementally.
    Each node reads what it needs and returns a dict of fields to update.
    """

    # ── Input ──────────────────────────────────────────────────────────────
    question : str          # the original user question (set at graph entry)

    # ── Routing ────────────────────────────────────────────────────────────
    route             : str     # "sql" | "rag" | "hybrid"
    route_confidence  : float   # 0.0 – 1.0
    route_reasoning   : str     # LLM's explanation of routing decision

    # ── Agent results (populated by agent nodes) ───────────────────────────
    sql_result : Optional[Any]  # SQLAgentResult  (from sql_agent.py)
    rag_result : Optional[Any]  # RAGAgentResult  (from rag_agent.py)

    # ── Final output (populated by synthesis node) ─────────────────────────
    final_response : Optional[Any]   # FinalResponse (from synthesis_agent.py)

    # ── Error tracking ─────────────────────────────────────────────────────
    errors : list[str]       # list of non-fatal errors encountered
