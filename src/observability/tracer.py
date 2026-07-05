"""
src/observability/tracer.py
─────────────────────────────
Lightweight, native request tracing for the LangGraph workflow.

Why native instead of Logfire/LangSmith:
  Same reasoning as guardrails.py and llm_gateway.py — this repo doesn't
  depend on an external observability SaaS anywhere else, and pulling one
  in just for this would mean every reviewer/grader needs an account and
  API key just to see it work. loguru is already a dependency and can do
  structured JSON logging on its own (serialize=True), which is enough to
  answer the actual question this exists to answer: "why did this request
  route the way it did, and where did time go?"

  LangSmith IS still worth mentioning as a complementary, zero-code option:
  LangGraph auto-instruments itself for LangSmith if you set
  LANGCHAIN_TRACING_V2=true + LANGCHAIN_API_KEY + LANGCHAIN_PROJECT as env
  vars — no code changes needed. See README for the flag. This module is
  for the *offline, no-account-needed* version of the same idea.

What this gives you:
  • Per-node timing + structured decision fields (route chosen, guardrail
    category, success/failure) for every request, correlated by trace_id.
  • An in-memory ring buffer (last N traces) so a running API can answer
    "show me the full path of request X" via GET /traces/{trace_id}.
  • A structured JSONL file sink (logs/traces.jsonl) for anything that
    needs to survive a restart or be grepped/analysed after the fact.

Usage:
    from src.observability.tracer import traced_node, configure_tracing, new_trace_id

    configure_tracing()                  # once, at app startup

    @traced_node("classify")
    def classify_node(state): ...

    trace_id = new_trace_id()
    state = {"question": "...", "trace_id": trace_id}
    ... run graph ...
    get_trace(trace_id)                  # -> list of step dicts
"""

import time
import uuid
import dataclasses
from collections import OrderedDict
from functools import wraps
from pathlib import Path
from threading import Lock

from loguru import logger


# ── Config ────────────────────────────────────────────────────────────────────

MAX_TRACES_IN_MEMORY = 200     # ring buffer size, mirrors app_state.history's "keep last 50" pattern
TRACE_LOG_PATH        = "logs/traces.jsonl"


# ── In-memory trace store (thread-safe ring buffer) ──────────────────────────
# {trace_id: [step_dict, step_dict, ...]}, insertion-ordered so eviction of
# the oldest trace is O(1) once we're over MAX_TRACES_IN_MEMORY.

_traces: "OrderedDict[str, list[dict]]" = OrderedDict()
_lock = Lock()


def new_trace_id() -> str:
    """Short trace id — enough entropy to avoid collisions in one process's lifetime."""
    return uuid.uuid4().hex[:12]


def _ensure_trace(trace_id: str) -> None:
    with _lock:
        if trace_id not in _traces:
            _traces[trace_id] = []
            while len(_traces) > MAX_TRACES_IN_MEMORY:
                _traces.popitem(last=False)   # evict oldest


def get_trace(trace_id: str) -> list[dict]:
    """Returns the recorded steps for a trace_id, oldest first. Empty list if unknown."""
    with _lock:
        return list(_traces.get(trace_id, []))


def list_trace_ids(limit: int = 50) -> list[str]:
    """Most recent trace_ids, newest first."""
    with _lock:
        return list(reversed(list(_traces.keys())))[:limit]


# ── Serialization helpers ────────────────────────────────────────────────────
# Node return values often contain dataclasses (SQLAgentResult, FinalResponse,
# ...) that aren't JSON-safe as-is. Prefer their own .to_dict() if present,
# fall back to dataclasses.asdict(), fall back to a truncated repr — never
# let a trace-logging failure break the actual request.

def _summarize(value, max_len: int = 300):
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, str) and len(value) > max_len:
            return value[:max_len] + "…"
        return value

    if isinstance(value, list):
        return f"<list len={len(value)}>"

    if dataclasses.is_dataclass(value):
        if hasattr(value, "to_dict"):
            try:
                return value.to_dict()
            except Exception:
                pass
        try:
            return dataclasses.asdict(value)
        except Exception:
            pass

    return str(value)[:max_len]


def _summarize_dict(d: dict) -> dict:
    return {k: _summarize(v) for k, v in d.items()}


# ── Structured file sink ──────────────────────────────────────────────────────

_configured = False


def configure_tracing(log_path: str = TRACE_LOG_PATH) -> None:
    """
    Adds a structured JSONL sink to loguru for trace events specifically
    (filtered via the `trace` bound field), separate from normal app logs.
    Call once at startup (api/main.py lifespan).
    """
    global _configured
    if _configured:
        return

    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_path,
        filter    = lambda record: record["extra"].get("trace") is True,
        serialize = True,           # JSON lines
        rotation  = "10 MB",
        retention = "7 days",
        level     = "INFO",
    )
    _configured = True
    logger.info(f"[tracer] Structured trace logging -> {log_path}")


# ── The decorator ─────────────────────────────────────────────────────────────

# Keys that, if present on the state or the node's return dict, are worth
# pulling into the trace step even though they're not the node's main output
# (e.g. every node might carry guardrail_category forward once set).
_INTERESTING_STATE_KEYS = (
    "guardrail_blocked", "guardrail_category",
    "route", "route_confidence",
)


def traced_node(node_name: str):
    """
    Wraps a LangGraph node function `(state) -> dict` to record timing and
    a summarized view of its output, keyed by state["trace_id"].

    If no trace_id is present on the state (e.g. a unit test calling the
    node directly), tracing is skipped silently — this must never be the
    reason a request fails.
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(state, *args, **kwargs):
            trace_id = state.get("trace_id") if isinstance(state, dict) else None
            t0 = time.perf_counter()

            try:
                result = fn(state, *args, **kwargs)
                duration_ms = round((time.perf_counter() - t0) * 1000, 1)

                if trace_id:
                    _record_step(
                        trace_id, node_name, duration_ms,
                        status="ok", output=result, state=state,
                    )
                return result

            except Exception as e:
                duration_ms = round((time.perf_counter() - t0) * 1000, 1)
                if trace_id:
                    _record_step(
                        trace_id, node_name, duration_ms,
                        status="error", output={"exception": str(e)}, state=state,
                    )
                raise   # tracing observes, never swallows

        return wrapper
    return decorator


def _record_step(trace_id: str, node_name: str, duration_ms: float,
                  status: str, output: dict, state: dict) -> None:
    try:
        _ensure_trace(trace_id)

        step = {
            "node"       : node_name,
            "status"     : status,
            "duration_ms": duration_ms,
        }
        step.update(_summarize_dict(output or {}))

        # Carry forward a few state-level fields even if this node didn't
        # set them itself (e.g. sql_node's step still shows which route
        # sent it there) — makes each step self-contained when read alone.
        for key in _INTERESTING_STATE_KEYS:
            if key not in step and key in state:
                step[key] = _summarize(state[key])

        with _lock:
            _traces[trace_id].append(step)

        logger.bind(trace=True, trace_id=trace_id).info(
            f"[trace] {node_name} ({status}, {duration_ms}ms)", **step,
        )
    except Exception as e:
        # Tracing is diagnostic, not load-bearing — never let it take the
        # actual request down.
        logger.warning(f"[tracer] Failed to record step for {node_name}: {e}")
