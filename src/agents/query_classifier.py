"""
src/agents/query_classifier.py
────────────────────────────────
Classifies an incoming question into one of three routing categories:
  • SQL     — needs structured ERP data (vendors, POs, invoices, spend)
  • RAG     — needs unstructured documents (contracts, policies)
  • HYBRID  — needs both sources combined

This is Agent 1 in the LangGraph workflow.

Design notes:
  - Uses Claude Haiku with temperature=0 for deterministic routing
  - Returns a structured RouteDecision object, not raw text
  - Includes confidence score and reasoning (useful for debugging + RAGAS eval)

Usage:
    from src.agents.query_classifier import classify_query, RouteDecision
    decision = classify_query("Which vendors have open POs above 50K?")
    print(decision.route)      # "sql"
    print(decision.reasoning)  # "Requires querying purchase_orders table..."
"""

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from src.agents.bedrock_client import get_llm


# ── Route types ───────────────────────────────────────────────────────────────

class Route(str, Enum):
    SQL    = "sql"
    RAG    = "rag"
    HYBRID = "hybrid"


@dataclass
class RouteDecision:
    route      : Route
    confidence : float          # 0.0 – 1.0
    reasoning  : str            # LLM's explanation of its choice
    raw_query  : str            # original question


# ── System prompt ─────────────────────────────────────────────────────────────

CLASSIFIER_SYSTEM_PROMPT = """You are a query routing agent for a procurement intelligence system.

Your job is to classify incoming questions into exactly one of three categories:

CATEGORY DEFINITIONS:
─────────────────────
SQL   → The question requires querying structured ERP database tables:
        - Vendor information (names, countries, categories, ratings, onboarding dates)
        - Purchase Orders (amounts, status, dates, categories)
        - Invoices (amounts, due dates, payment status, overdue items)
        - Spend analysis (total spend by category, monthly trends, averages)
        Examples: totals, counts, lists, comparisons of numeric data, date ranges

RAG   → The question requires retrieving information from unstructured documents:
        - Vendor contract clauses (payment terms, penalty clauses, termination, force majeure)
        - Procurement policy rules (approval thresholds, single-source rules, conflict of interest)
        Examples: "what does the contract say about...", "what is our policy for..."

HYBRID → The question requires BOTH structured ERP data AND document content:
        - Combines a database query with a contract/policy lookup
        Examples: "which vendors have open POs above 50K AND no penalty clause"
                  "what is our IT spend AND what does policy say about IT thresholds"

IMPORTANT RULES:
  - If the question mentions vendors by name AND asks about contract terms → HYBRID
  - If the question asks for numbers/amounts/counts/lists → SQL (or HYBRID if also asks about clauses)
  - If the question asks "what does...say" or "is there a clause" → RAG
  - When in doubt between SQL and HYBRID, choose HYBRID (safer)

Respond ONLY with a JSON object in this exact format (no markdown, no preamble):
{
  "route": "sql" | "rag" | "hybrid",
  "confidence": 0.0-1.0,
  "reasoning": "one sentence explaining the choice"
}"""


# ── Classifier ────────────────────────────────────────────────────────────────

def classify_query(query: str) -> RouteDecision:
    """
    Classify a natural language query into SQL, RAG, or HYBRID.

    Args:
        query : The user's question.

    Returns:
        RouteDecision with route, confidence, and reasoning.
    """
    logger.info(f"Classifying query: '{query[:80]}...' " if len(query) > 80 else f"Classifying: '{query}'")

    llm = get_llm(temperature=0.0, max_tokens=256)

    messages = [
        SystemMessage(content=CLASSIFIER_SYSTEM_PROMPT),
        HumanMessage(content=f"Classify this procurement query:\n\n{query}"),
    ]

    try:
        response = llm.invoke(messages)
        raw      = response.content.strip()

        # Strip markdown fences if present
        raw = re.sub(r"```json|```", "", raw).strip()

        parsed = json.loads(raw)

        decision = RouteDecision(
            route      = Route(parsed["route"].lower()),
            confidence = float(parsed.get("confidence", 0.9)),
            reasoning  = parsed.get("reasoning", ""),
            raw_query  = query,
        )

        logger.success(
            f"  Route: {decision.route.value.upper()} "
            f"(confidence={decision.confidence:.2f}) — {decision.reasoning}"
        )
        return decision

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning(f"  Classifier parse error: {e}. Raw response: '{raw}'. Defaulting to HYBRID.")
        return RouteDecision(
            route      = Route.HYBRID,
            confidence = 0.5,
            reasoning  = f"Parse error — defaulted to hybrid for safety. Error: {e}",
            raw_query  = query,
        )


# ── Rule-based fallback (no LLM needed for obvious cases) ────────────────────

_SQL_KEYWORDS = [
    "total", "count", "sum", "average", "how many", "list all",
    "top ", "show all", "overdue by", "grouped by", "this quarter",
    "last month", "in 2024", "in 2023", "vendor rating", "onboarded",
]

_RAG_KEYWORDS = [
    "contract say", "policy say", "clause", "penalty", "termination",
    "payment terms", "force majeure", "what does", "is there a",
    "according to", "under the contract", "compliance", "policy for",
    "single-source", "sole source", "conflict of interest",
]

def classify_query_fast(query: str) -> RouteDecision | None:
    """
    Rule-based classifier for obvious cases — skips LLM call entirely.
    Returns None if ambiguous (fall through to LLM classifier).
    Used as a pre-filter to reduce Bedrock API calls.
    """
    q_lower = query.lower()

    has_sql = any(kw in q_lower for kw in _SQL_KEYWORDS)
    has_rag = any(kw in q_lower for kw in _RAG_KEYWORDS)

    if has_sql and has_rag:
        return RouteDecision(Route.HYBRID, 0.85,
                             "Contains both data and document keywords.", query)
    if has_sql and not has_rag:
        return RouteDecision(Route.SQL, 0.85,
                             "Contains structured data keywords.", query)
    if has_rag and not has_sql:
        return RouteDecision(Route.RAG, 0.85,
                             "Contains document/clause keywords.", query)
    return None     # ambiguous — let LLM decide


def route_query(query: str, use_fast_path: bool = True) -> RouteDecision:
    """
    Main entry point. Tries fast rule-based routing first,
    falls back to LLM classification for ambiguous queries.

    Args:
        query          : User's question.
        use_fast_path  : If True, tries rule-based routing first.

    Returns:
        RouteDecision
    """
    if use_fast_path:
        fast = classify_query_fast(query)
        if fast is not None:
            logger.info(f"  Fast-path route: {fast.route.value.upper()}")
            return fast

    return classify_query(query)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_queries = [
        "What is our total open PO value for IT Services vendors this quarter?",
        "Does the vendor contract include a late delivery penalty clause?",
        "Which vendors have open POs above 50K AND no penalty clause in their contract?",
        "What is our policy for single-source procurement above 100K?",
        "Show all invoices overdue by more than 30 days grouped by category.",
        "What are the payment terms in the contract and how much do we owe vendor X?",
    ]

    print(f"\n{'='*60}")
    print("QUERY CLASSIFIER TEST")
    print(f"{'='*60}")

    for q in test_queries:
        decision = route_query(q)
        print(f"\nQ: {q}")
        print(f"   Route      : {decision.route.value.upper()}")
        print(f"   Confidence : {decision.confidence:.2f}")
        print(f"   Reasoning  : {decision.reasoning}")
