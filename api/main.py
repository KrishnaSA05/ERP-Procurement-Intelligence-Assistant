"""
api/main.py
────────────
FastAPI backend for the ERP Procurement Intelligence Assistant.

Endpoints:
  POST /query        — run a procurement question through the full pipeline
  GET  /health       — check DB + ChromaDB connectivity
  GET  /history      — last N queries with answers
  GET  /docs         — auto-generated Swagger UI (built into FastAPI)

Run locally:
    uvicorn api.main:app --reload --port 8000

Then open: http://localhost:8000/docs
"""

import sys
import time
from pathlib import Path
from datetime import datetime, timezone
from contextlib import asynccontextmanager

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from api.models import (
    QueryRequest, QueryResponse, CitationModel,
    HealthResponse, HistoryItem, HistoryResponse,
)
from src.graph.langgraph_workflow import build_workflow, run_query
from src.data.db_loader           import check_connection
from src.vectorstore.chroma_store import ChromaStore


# ── App state (shared across requests) ───────────────────────────────────────

class AppState:
    workflow = None
    store    = None
    history  : list[dict] = []
    query_counter : int = 0

app_state = AppState()


# ── Lifespan: build workflow once at startup ──────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise heavy resources once at startup, clean up on shutdown."""
    logger.info("Starting up — building LangGraph workflow...")
    try:
        app_state.store    = ChromaStore()
        app_state.workflow = build_workflow(chroma_store=app_state.store)
        logger.success("✓ Workflow ready")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        # Don't crash — let health endpoint report degraded status
    yield
    logger.info("Shutting down...")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "ERP Procurement Intelligence API",
    description = (
        "Agentic RAG system that answers procurement questions by routing "
        "between a SQL agent (ERP data) and a RAG agent (vendor contracts + policies)."
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
)

# CORS — allow Streamlit (localhost:8501) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["http://localhost:8501", "http://127.0.0.1:8501", "*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return {
        "service" : "ERP Procurement Intelligence API",
        "version" : "1.0.0",
        "docs"    : "/docs",
        "health"  : "/health",
    }


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Run a natural language procurement question through the full pipeline.

    Routes automatically between:
    - **SQL Agent** for structured ERP data (vendors, POs, invoices, spend)
    - **RAG Agent** for unstructured documents (contracts, policies)
    - **Hybrid** when both are needed

    Returns the answer, source citations, SQL used, and raw data rows.
    """
    if app_state.workflow is None:
        raise HTTPException(
            status_code = 503,
            detail      = "Workflow not initialised. Check /health for details.",
        )

    question = request.question.strip()
    t_start  = time.time()

    logger.info(f"POST /query — '{question[:80]}'")

    try:
        # Run through LangGraph
        state      = run_query(app_state.workflow, question)
        latency_ms = round((time.time() - t_start) * 1000, 1)

        fr = state.get("final_response")
        if not fr:
            raise HTTPException(status_code=500, detail="No response generated.")

        # Build response
        citations = [
            CitationModel(
                source_file = c.source_file,
                doc_type    = c.doc_type,
                page_number = c.page_number,
                similarity  = round(c.similarity, 4),
                excerpt     = c.excerpt,
            )
            for c in (fr.citations or [])
        ]

        response = QueryResponse(
            question     = question,
            final_answer = fr.final_answer,
            route_used   = fr.route_used,
            sql_query    = fr.sql_query or None,
            citations    = citations,
            data_rows    = fr.data_rows or [],
            success      = fr.success,
            error        = fr.error or None,
            latency_ms   = latency_ms,
        )

        # Store in history (keep last 50)
        app_state.query_counter += 1
        app_state.history.append({
            "id"            : app_state.query_counter,
            "question"      : question,
            "route_used"    : fr.route_used,
            "answer_preview": fr.final_answer[:120],
            "timestamp"     : datetime.now(timezone.utc).isoformat(),
            "success"       : fr.success,
        })
        if len(app_state.history) > 50:
            app_state.history = app_state.history[-50:]

        logger.success(f"  ✓ Response in {latency_ms}ms | route={fr.route_used.upper()}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"  Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health_endpoint():
    """
    Check connectivity to PostgreSQL and ChromaDB.
    Returns row counts and collection sizes.
    """
    # Check PostgreSQL
    db_status = check_connection()

    # Check ChromaDB
    try:
        stats        = app_state.store.collection_stats() if app_state.store else {}
        chroma_status = {"status": "ok", "collections": stats}
    except Exception as e:
        chroma_status = {"status": "error", "detail": str(e)}

    overall = "ok"
    if db_status.get("status") != "ok":
        overall = "degraded"
    if chroma_status.get("status") != "ok":
        overall = "degraded"
    if app_state.workflow is None:
        overall = "degraded"

    return HealthResponse(
        status      = overall,
        database    = db_status,
        vectorstore = chroma_status,
        timestamp   = datetime.now(timezone.utc).isoformat(),
    )


@app.get("/history", response_model=HistoryResponse)
async def history_endpoint(limit: int = 20):
    """
    Return the last N queries with answer previews.
    Useful for the Streamlit sidebar and debugging.
    """
    limit = min(limit, 50)
    items = app_state.history[-limit:][::-1]   # most recent first

    return HistoryResponse(
        items = [HistoryItem(**item) for item in items],
        total = len(app_state.history),
    )


@app.get("/sample-questions")
async def sample_questions():
    """Return sample questions for the UI quick-select."""
    return {
        "questions": [
            {
                "text"    : "What is our total open PO value for IT Services this quarter?",
                "route"   : "sql",
                "category": "Spend Analysis",
            },
            {
                "text"    : "Show all invoices overdue by more than 30 days grouped by category.",
                "route"   : "sql",
                "category": "Invoice Management",
            },
            {
                "text"    : "Who are our top 5 vendors by total PO value in 2024?",
                "route"   : "sql",
                "category": "Vendor Analysis",
            },
            {
                "text"    : "Does our vendor contract include a late delivery penalty clause?",
                "route"   : "rag",
                "category": "Contract Review",
            },
            {
                "text"    : "What is our policy for single-source procurement above $100,000?",
                "route"   : "rag",
                "category": "Policy Compliance",
            },
            {
                "text"    : "What are the payment terms in the standard vendor contract?",
                "route"   : "rag",
                "category": "Contract Review",
            },
            {
                "text"    : "Which vendors have open POs above $50,000 and what do their contracts say about late delivery penalties?",
                "route"   : "hybrid",
                "category": "Hybrid Intelligence",
            },
            {
                "text"    : "What is our total IT spend this year and what does policy say about IT procurement thresholds?",
                "route"   : "hybrid",
                "category": "Hybrid Intelligence",
            },
        ]
    }


# ── Error handlers ────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code = 500,
        content     = {"detail": "Internal server error", "error": str(exc)},
    )


# ── Dev runner ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host    = "0.0.0.0",
        port    = 8000,
        reload  = True,
        log_level = "info",
    )
