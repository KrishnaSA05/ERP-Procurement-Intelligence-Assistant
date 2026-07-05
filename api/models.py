"""
api/models.py

Pydantic models for FastAPI request validation and response serialisation.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# ── Request ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question : str = Field(
        ...,
        min_length = 5,
        max_length = 500,
        description = "Natural language procurement question",
        examples   = ["Which vendors have open POs above $50,000?"],
    )


# ── Sub-models ────────────────────────────────────────────────────────────────

class CitationModel(BaseModel):
    source_file : str
    doc_type    : str
    page_number : int
    similarity  : float
    excerpt     : str


class QueryResponse(BaseModel):
    question     : str
    final_answer : str
    route_used   : str                          # "sql" | "rag" | "hybrid" | "blocked"
    sql_query    : Optional[str]   = None       # shown for transparency
    citations    : list[CitationModel] = []
    data_rows    : list[dict]          = []
    success      : bool = True
    error        : Optional[str]  = None
    latency_ms   : Optional[float] = None       # response time
    guardrail_blocked  : bool           = False  # True if the request was gated
    guardrail_category : Optional[str]  = None   # "off_topic" | "jailbreak" | "unsafe"
    trace_id           : Optional[str]  = None   # look up full step trace via GET /traces/{trace_id}


class HealthResponse(BaseModel):
    status      : str                           # "ok" | "degraded" | "error"
    database    : dict
    vectorstore : dict
    timestamp   : str


class HistoryItem(BaseModel):
    id           : int
    question     : str
    route_used   : str
    answer_preview: str                         # first 120 chars
    timestamp    : str
    success      : bool


class HistoryResponse(BaseModel):
    items : list[HistoryItem]
    total : int
