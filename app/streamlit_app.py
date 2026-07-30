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
import base64
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
#API_URL = os.environ.get("API_URL", "http://api:8000") Used when deployed online


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
.answer-card   { background:#1e2130; color:#e8eaf0 !important;
                 border-left:4px solid #1a6de0;
                 padding:16px 20px; border-radius:4px; margin:8px 0; }
.answer-card * { color:#e8eaf0 !important; }

/* Citation card */
.citation-card { background:#252839; color:#c8cad4 !important;
                 border:1px solid #3a3f55;
                 padding:10px 14px; border-radius:4px; margin:4px 0;
                 font-size:13px; }
.citation-card * { color:#c8cad4 !important; }

/* Metric tile */
.metric-tile   { text-align:center; }

/* Health dot */
.dot-green { color:#28a745; font-size:18px; }
.dot-red   { color:#dc3545; font-size:18px; }
.dot-amber { color:#ffc107; font-size:18px; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────

if "messages"      not in st.session_state: st.session_state.messages      = []
if "query_count"   not in st.session_state: st.session_state.query_count   = 0
if "total_latency" not in st.session_state: st.session_state.total_latency = 0.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def call_api(question: str, image_bytes: bytes = None) -> dict | None:
    """POST /query to FastAPI backend. Attaches a base64 image if provided
    (an invoice/PO photo — see VisionAgent)."""
    try:
        payload = {"question": question}
        if image_bytes:
            payload["image_base64"] = base64.b64encode(image_bytes).decode("utf-8")

        resp = requests.post(
            f"{API_URL}/query",
            json    = payload,
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


# ── Answer renderer ───────────────────────────────────────────────────────────
# FIX: defined here (before first use) so it is available when the chat
#      history loop replays previous messages on every Streamlit re-run.

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

    # Vision extraction (only present if an image was attached)
    vision = data.get("vision_extracted")
    if vision:
        with st.expander("🧾 Extracted from uploaded image", expanded=not vision.get("success", True) is False):
            if not vision.get("success"):
                st.warning(f"Vision extraction failed: {vision.get('error', 'unknown error')}")
            else:
                vcol1, vcol2 = st.columns(2)
                with vcol1:
                    st.text(f"Document type : {vision.get('document_type', '')}")
                    st.text(f"Vendor        : {vision.get('vendor_name', '')}")
                    st.text(f"PO number     : {vision.get('po_number', '')}")
                with vcol2:
                    st.text(f"Invoice #     : {vision.get('invoice_number', '')}")
                    st.text(f"Date          : {vision.get('invoice_date', '')}")
                    st.text(f"Total         : {vision.get('total_amount', '')}")
                line_items = vision.get("line_items") or []
                if line_items:
                    import pandas as pd
                    st.dataframe(pd.DataFrame(line_items), use_container_width=True, hide_index=True)

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
# FIX: _render_answer is now defined above, so this loop works correctly.

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            # Assistant message — render full response card.
            # FIX: _render_answer() has no return value (implicitly None).
            # Calling it as a bare top-level statement made Streamlit's
            # "magic commands" feature auto-write that None to the page —
            # that's the stray "None" that was appearing under every
            # response. Assigning to a throwaway variable suppresses it.
            data = msg.get("data", {})
            if data:
                _ = _render_answer(data)
            else:
                _ = st.markdown(msg["content"])


# ── Input ─────────────────────────────────────────────────────────────────────

# Optional invoice/PO photo — extracted by the Vision Agent and folded into
# the question before routing (e.g. "does this match our records?").
uploaded_image = st.file_uploader(
    "📎 Attach an invoice/PO photo (optional)",
    type = ["png", "jpg", "jpeg"],
    key  = "invoice_uploader",
)
if uploaded_image is not None:
    st.image(uploaded_image, caption="Attached — will be sent with your next question", width=200)

# Pre-fill from sidebar sample click
prefill = st.session_state.pop("prefill_question", "")

question = st.chat_input(
    "Ask a procurement question...",
    key = "chat_input",
) or prefill

if question:
    question = question.strip()
    image_bytes = uploaded_image.getvalue() if uploaded_image is not None else None

    # Show user message
    with st.chat_message("user"):
        st.markdown(question)
        if image_bytes:
            st.image(image_bytes, width=160)
    st.session_state.messages.append({"role": "user", "content": question})

    # Call API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            t0      = time.time()
            data    = call_api(question, image_bytes=image_bytes)
            elapsed = round((time.time() - t0) * 1000)

        if data:
            if not data.get("latency_ms"):
                data["latency_ms"] = elapsed

            # FIX: same magic-command gotcha as the history replay loop above —
            # assign the (None) return value instead of leaving it bare.
            _ = _render_answer(data)

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
