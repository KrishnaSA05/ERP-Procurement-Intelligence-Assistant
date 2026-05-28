"""
app/streamlit_app.py
─────────────────────
Streamlit frontend for the ERP Procurement Intelligence Assistant.

Features:
  • Chat-style Q&A interface
  • Route badge (SQL / RAG / Hybrid) per answer
  • Collapsible SQL query display
  • Source citations with similarity scores
  • Raw data table (expandable)
  • Query history sidebar
  • Sample questions quick-select
  • System health indicator

Run:
    streamlit run app/streamlit_app.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import requests
import streamlit as st
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title = "Procurement Intelligence",
    page_icon  = "🔍",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

API_URL = "http://localhost:8000"


# ── Styling ───────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Route badges */
.badge-sql     { background:#1a6de0; color:white; padding:3px 10px;
                 border-radius:12px; font-size:12px; font-weight:600; }
.badge-rag     { background:#28a745; color:white; padding:3px 10px;
                 border-radius:12px; font-size:12px; font-weight:600; }
.badge-hybrid  { background:#7b2dbd; color:white; padding:3px 10px;
                 border-radius:12px; font-size:12px; font-weight:600; }

/* Answer card */
.answer-card   { background:#f8f9fa; border-left:4px solid #1a6de0;
                 padding:16px 20px; border-radius:4px; margin:8px 0; }

/* Citation card */
.citation-card { background:#ffffff; border:1px solid #dee2e6;
                 padding:10px 14px; border-radius:4px; margin:4px 0;
                 font-size:13px; }

/* Metric tile */
.metric-tile   { text-align:center; }

/* Health dot */
.dot-green { color:#28a745; font-size:18px; }
.dot-red   { color:#dc3545; font-size:18px; }
.dot-amber { color:#ffc107; font-size:18px; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────

if "messages"     not in st.session_state: st.session_state.messages     = []
if "query_count"  not in st.session_state: st.session_state.query_count  = 0
if "total_latency"not in st.session_state: st.session_state.total_latency= 0.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def call_api(question: str) -> dict | None:
    """POST /query to FastAPI backend."""
    try:
        resp = requests.post(
            f"{API_URL}/query",
            json    = {"question": question},
            timeout = 120,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Make sure `uvicorn api.main:app` is running.")
        return None
    except requests.exceptions.Timeout:
        st.error("⏱ Request timed out (>120s). The LLM may be slow.")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def get_health() -> dict | None:
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        return resp.json()
    except Exception:
        return None


def get_sample_questions() -> list[dict]:
    try:
        resp = requests.get(f"{API_URL}/sample-questions", timeout=5)
        return resp.json().get("questions", [])
    except Exception:
        return []


def route_badge(route: str) -> str:
    badges = {
        "sql"   : '<span class="badge-sql">📊 ERP Database</span>',
        "rag"   : '<span class="badge-rag">📄 Documents</span>',
        "hybrid": '<span class="badge-hybrid">🔀 ERP + Documents</span>',
    }
    return badges.get(route, route)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔍 Procurement AI")
    st.caption("ERP + Contract Intelligence")
    st.divider()

    # ── Health check ──────────────────────────────────────────────────────
    st.subheader("System Status")
    health = get_health()
    if health:
        status = health.get("status", "unknown")
        dot    = "🟢" if status == "ok" else ("🟡" if status == "degraded" else "🔴")
        st.markdown(f"{dot} **Overall:** {status.upper()}")

        db_ok = health.get("database", {}).get("status") == "ok"
        vs_ok = health.get("vectorstore", {}).get("status") == "ok"
        st.markdown(f"{'🟢' if db_ok else '🔴'} PostgreSQL")
        st.markdown(f"{'🟢' if vs_ok else '🔴'} ChromaDB")

        # Row counts
        row_counts = health.get("database", {}).get("row_counts", {})
        if row_counts:
            with st.expander("Database rows"):
                for table, count in row_counts.items():
                    st.text(f"  {table}: {count}")

        # Vector counts
        cols = health.get("vectorstore", {}).get("collections", {})
        if cols:
            with st.expander("Vector chunks"):
                for col, count in cols.items():
                    st.text(f"  {col}: {count}")
    else:
        st.markdown("🔴 API unreachable")

    st.divider()

    # ── Session metrics ───────────────────────────────────────────────────
    st.subheader("Session Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Queries", st.session_state.query_count)
    with col2:
        avg_ms = (
            round(st.session_state.total_latency / st.session_state.query_count)
            if st.session_state.query_count > 0 else 0
        )
        st.metric("Avg ms", avg_ms)

    st.divider()

    # ── Sample questions ──────────────────────────────────────────────────
    st.subheader("Sample Questions")
    samples = get_sample_questions()

    if samples:
        categories = list({s["category"] for s in samples})
        for cat in categories:
            with st.expander(cat):
                for s in [q for q in samples if q["category"] == cat]:
                    route_col = {
                        "sql"   : "#1a6de0",
                        "rag"   : "#28a745",
                        "hybrid": "#7b2dbd",
                    }.get(s["route"], "#333")

                    if st.button(
                        s["text"],
                        key     = f"sample_{s['text'][:30]}",
                        help    = f"Route: {s['route'].upper()}",
                        use_container_width = True,
                    ):
                        st.session_state["prefill_question"] = s["text"]
                        st.rerun()

    st.divider()

    # ── Route legend ──────────────────────────────────────────────────────
    st.subheader("Route Types")
    st.markdown("📊 **SQL** — ERP data queries")
    st.markdown("📄 **RAG** — Contract & policy docs")
    st.markdown("🔀 **Hybrid** — Both combined")


# ── Main area ─────────────────────────────────────────────────────────────────

st.title("Procurement Intelligence Assistant")
st.caption(
    "Ask questions about vendors, purchase orders, invoices, "
    "contract clauses, and procurement policies."
)

# ── Chat history ──────────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            # Assistant message — render full response card
            data = msg.get("data", {})
            _render_answer(data) if data else st.markdown(msg["content"])


def _render_answer(data: dict):
    """Render a full answer card with badges, citations, SQL, and data table."""

    route = data.get("route_used", "")

    # Route badge + latency
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(route_badge(route), unsafe_allow_html=True)
    with col2:
        ms = data.get("latency_ms")
        if ms:
            st.caption(f"⏱ {ms:.0f}ms")

    # Main answer
    st.markdown(
        f'<div class="answer-card">{data.get("final_answer", "")}</div>',
        unsafe_allow_html=True,
    )

    # SQL transparency
    sql = data.get("sql_query")
    if sql:
        with st.expander("🔎 SQL Query Used"):
            st.code(sql, language="sql")

    # Data table
    rows = data.get("data_rows", [])
    if rows:
        with st.expander(f"📋 Data ({len(rows)} rows)"):
            import pandas as pd
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width = True,
                hide_index          = True,
            )

    # Citations
    citations = data.get("citations", [])
    if citations:
        with st.expander(f"📚 Sources ({len(citations)} documents)"):
            for i, c in enumerate(citations, 1):
                sim_pct = int(c.get("similarity", 0) * 100)
                sim_bar = "█" * (sim_pct // 10) + "░" * (10 - sim_pct // 10)

                st.markdown(
                    f'<div class="citation-card">'
                    f'<b>[{i}] {c.get("source_file","")}</b> '
                    f'— page {c.get("page_number","")} '
                    f'— {c.get("doc_type","").upper()}<br>'
                    f'<span style="color:#888;font-size:11px">'
                    f'Relevance: {sim_bar} {sim_pct}%</span><br>'
                    f'<i>"{c.get("excerpt","")[:180]}..."</i>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# Patch session rendering to use _render_answer
for msg in st.session_state.messages:
    pass  # already rendered above, _render_answer available from here


# ── Input ─────────────────────────────────────────────────────────────────────

# Pre-fill from sidebar sample click
prefill = st.session_state.pop("prefill_question", "")

question = st.chat_input(
    "Ask a procurement question...",
    key = "chat_input",
) or prefill

if question:
    question = question.strip()

    # Show user message
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # Call API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            t0   = time.time()
            data = call_api(question)
            elapsed = round((time.time() - t0) * 1000)

        if data:
            if not data.get("latency_ms"):
                data["latency_ms"] = elapsed

            _render_answer(data)

            st.session_state.messages.append({
                "role"   : "assistant",
                "content": data.get("final_answer", ""),
                "data"   : data,
            })

            st.session_state.query_count   += 1
            st.session_state.total_latency += data.get("latency_ms", 0)

        else:
            error_msg = "Sorry, I couldn't process that question. Please check the API is running."
            st.error(error_msg)
            st.session_state.messages.append({
                "role"   : "assistant",
                "content": error_msg,
            })

# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Built with LangGraph · LangChain · Amazon Bedrock (Claude Haiku) · "
    "ChromaDB · PostgreSQL · FastAPI · Streamlit"
)
