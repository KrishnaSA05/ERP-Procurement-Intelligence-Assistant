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
                  "summarize our IT Services spending and any relevant contract obligations"
                  "how much do we spend on logistics and what are the contract payment terms"

IMPORTANT RULES:
  - If the question mentions vendors by name AND asks about contract terms → HYBRID
  - If the question asks for numbers/amounts/counts/lists → SQL (or HYBRID if also asks about clauses)
  - If the question asks "what does...say" or "is there a clause" → RAG
  - If the question asks about BOTH spending/amounts AND contracts/obligations/policy → HYBRID
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

    llm = get_llm(temperature=0.0, max_tokens=256, label="classifier")

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

# FIX: SQL keywords are now whole-word patterns (not substrings).
# "sum" was matching "summarize"; "total" was fine but "count" could match
# "account". Using \b word boundaries via re.search() eliminates false hits.
_SQL_WORD_KEYWORDS = [
    r"\btotal\b", r"\bcount\b", r"\bhow many\b", r"\bhow much\b",
    r"\blist all\b", r"\btop \d", r"\bshow all\b", r"\boverdue\b",
    r"\bgrouped by\b", r"\bthis quarter\b", r"\blast month\b",
    r"\blast quarter\b", r"\bin 2024\b", r"\bin 2023\b",
    r"\bvendor rating\b", r"\bonboarded\b", r"\bspend\b",
    r"\bspending\b", r"\bexpenditure\b", r"\bpurchase order\b",
    r"\bopen po\b", r"\bpo value\b", r"\bpo count\b", r"\baverage po\b",
    r"\binvoice\b", r"\b\d+k\b", r"\bbudget\b", r"\bamount\b",
]

# FIX: RAG keywords now include standalone "contract", "policy", "obligation"
# so hybrid queries like "...contract obligations" are caught correctly.
_RAG_WORD_KEYWORDS = [
    r"\bcontract\b", r"\bpolicy\b", r"\bobligation\b", r"\bclause\b",
    r"\bpenalty\b", r"\btermination\b", r"\bpayment terms\b",
    r"\bforce majeure\b", r"\bwhat does\b", r"\bwhat do.*say\b",
    r"\bis there a\b", r"\baccording to\b", r"\bunder the contract\b",
    r"\bcompliance\b", r"\bprocurement rule\b", r"\bsingle.source\b",
    r"\bsole source\b", r"\bconflict of interest\b", r"\bsla\b",
    r"\bindemnif\b", r"\bliabilit\b", r"\bwarranty\b",
]


def _matches_any(patterns: list[str], text: str) -> bool:
    """Return True if any regex pattern matches in text (case-insensitive)."""
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def classify_query_fast(query: str) -> RouteDecision | None:
    """
    Rule-based classifier for obvious cases — skips LLM call entirely.
    Returns None if ambiguous (fall through to LLM classifier).
    Used as a pre-filter to reduce API calls.

    FIX: Uses regex word-boundary matching instead of plain substring checks
    to prevent false positives like "sum" matching "summarize".
    """
    has_sql = _matches_any(_SQL_WORD_KEYWORDS, query)
    has_rag = _matches_any(_RAG_WORD_KEYWORDS, query)

    if has_sql and has_rag:
        return RouteDecision(Route.HYBRID, 0.85,
                             "Contains both structured-data and document keywords.", query)
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
        "Summarize our IT Services spending and any relevant contract obligations",
        "How much do we spend on logistics and what are the contract payment terms?",
        "Which vendor category has the highest average PO value?",
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
