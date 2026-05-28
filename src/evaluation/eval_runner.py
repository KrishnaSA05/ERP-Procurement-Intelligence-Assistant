"""
src/evaluation/eval_runner.py
──────────────────────────────
CLI evaluation runner with formatted report.

Runs a quick sanity check on each agent independently before
running the full RAGAS suite — so you know exactly which layer
is underperforming if scores are low.

Usage:
    # Full evaluation (all routes)
    python src/evaluation/eval_runner.py

    # RAG only (faster, most relevant for RAGAS)
    python src/evaluation/eval_runner.py --routes rag hybrid

    # Quick smoke test (3 questions, no RAGAS)
    python src/evaluation/eval_runner.py --smoke-test
"""

import sys
import json
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from loguru import logger


# ── Smoke test ────────────────────────────────────────────────────────────────

def run_smoke_test():
    """
    Quick 3-question test to verify all agents are working
    before running the expensive full RAGAS evaluation.
    """
    from src.agents.sql_agent        import SQLAgent
    from src.agents.rag_agent        import RAGAgent
    from src.graph.langgraph_workflow import build_workflow, run_query

    print("\n" + "=" * 55)
    print("SMOKE TEST — verifying all agents")
    print("=" * 55)

    passed = 0
    failed = 0

    # ── Test 1: SQL Agent ──────────────────────────────────────────────────
    print("\n[1/3] SQL Agent — structured data query")
    try:
        agent  = SQLAgent()
        result = agent.run("How many vendors do we have in total?")
        assert result.success, f"SQL failed: {result.error}"
        assert result.narrative, "Empty narrative"
        assert result.row_count >= 0, "Negative row count"
        print(f"  ✓ SQL Agent OK — {result.row_count} rows, narrative: '{result.narrative[:80]}...'")
        passed += 1
    except Exception as e:
        print(f"  ✗ SQL Agent FAILED: {e}")
        failed += 1

    # ── Test 2: RAG Agent ──────────────────────────────────────────────────
    print("\n[2/3] RAG Agent — document retrieval")
    try:
        agent  = RAGAgent()
        result = agent.run("What are the payment terms in vendor contracts?")
        assert result.success or "could not find" in result.answer.lower(), \
            f"RAG failed: {result.error}"
        assert result.answer, "Empty answer"
        print(f"  ✓ RAG Agent OK — {result.chunks_used} chunks, {len(result.citations)} citations")
        print(f"    Answer: '{result.answer[:80]}...'")
        passed += 1
    except Exception as e:
        print(f"  ✗ RAG Agent FAILED: {e}")
        print("    Hint: run ingestion pipeline first: python src/ingestion/ingest_pipeline.py --all")
        failed += 1

    # ── Test 3: Full workflow (hybrid) ────────────────────────────────────
    print("\n[3/3] LangGraph Workflow — hybrid query")
    try:
        workflow = build_workflow()
        state    = run_query(
            workflow,
            "Which vendors have open POs above $50,000 and do contracts mention penalties?"
        )
        fr = state.get("final_response")
        assert fr and fr.success, f"Workflow failed: {fr.error if fr else 'No response'}"
        assert fr.final_answer, "Empty final answer"
        print(f"  ✓ Workflow OK — route={fr.route_used.upper()}")
        print(f"    Answer: '{fr.final_answer[:80]}...'")
        passed += 1
    except Exception as e:
        print(f"  ✗ Workflow FAILED: {e}")
        failed += 1

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"Smoke test: {passed}/3 passed, {failed}/3 failed")
    print(f"{'='*55}")

    if failed == 0:
        print("✓ All systems ready — safe to run full RAGAS evaluation")
        return True
    else:
        print("✗ Fix failures above before running full evaluation")
        return False


# ── Per-question results table ────────────────────────────────────────────────

def print_sample_table(samples):
    """Print a per-question results table."""
    print(f"\n{'─'*100}")
    print(f"{'#':<4} {'Route':<8} {'Latency':>8}  {'Q (truncated)':<50} {'Status'}")
    print(f"{'─'*100}")

    for i, s in enumerate(samples, 1):
        q_short = s.question[:48] + ".." if len(s.question) > 50 else s.question
        status  = "✓" if s.success else f"✗ {s.error[:30]}"
        print(
            f"{i:<4} {s.route.upper():<8} {s.latency_ms:>7.0f}ms  "
            f"{q_short:<50} {status}"
        )

    print(f"{'─'*100}")


# ── Metric bar chart (ASCII) ──────────────────────────────────────────────────

def print_metric_bars(results):
    """Render ASCII bar chart of metric scores."""
    print("\nMETRIC SCORES (bar chart)")
    print(f"{'─'*55}")

    metrics = {
        "Faithfulness     ": results.faithfulness,
        "Answer Relevancy ": results.answer_relevancy,
        "Context Precision": results.context_precision,
        "Context Recall   ": results.context_recall,
    }

    for name, score in metrics.items():
        filled = int(score * 40)
        empty  = 40 - filled
        bar    = "█" * filled + "░" * empty
        flag   = "✓" if score >= 0.80 else ("~" if score >= 0.65 else "✗")
        print(f"  {name}: [{bar}] {score:.4f} {flag}")

    print(f"{'─'*55}")
    print("  ✓ ≥ 0.80 (production ready)  ~ ≥ 0.65 (acceptable)  ✗ < 0.65 (needs tuning)")


# ── Recommendations ───────────────────────────────────────────────────────────

def print_recommendations(results):
    """Print tuning recommendations based on scores."""
    recs = []

    if results.faithfulness < 0.80:
        recs.append((
            "Faithfulness",
            results.faithfulness,
            "Answer contains content not grounded in retrieved chunks. "
            "Try: reduce temperature (rag_agent), add stricter RAG system prompt, "
            "increase n_final chunks passed to LLM."
        ))

    if results.answer_relevancy < 0.80:
        recs.append((
            "Answer Relevancy",
            results.answer_relevancy,
            "Answers are not addressing the question directly. "
            "Try: improve RAG system prompt clarity, "
            "check that query embeddings and chunk embeddings use the same model."
        ))

    if results.context_precision < 0.80:
        recs.append((
            "Context Precision",
            results.context_precision,
            "Retrieved chunks contain noise/irrelevant content. "
            "Try: reduce chunk_size (currently 800), increase score_threshold in chroma_store, "
            "improve re-ranking in rag_agent."
        ))

    if results.context_recall < 0.80:
        recs.append((
            "Context Recall",
            results.context_recall,
            "Relevant content is missing from retrieved chunks. "
            "Try: increase n_retrieve (currently 10), reduce chunk overlap, "
            "verify PDFs were ingested fully."
        ))

    if not recs:
        print("\n✓ All metrics above 0.80 — pipeline is production-ready.")
        return

    print("\nTUNING RECOMMENDATIONS")
    print(f"{'─'*55}")
    for metric, score, rec in recs:
        print(f"\n  [{metric}: {score:.4f}]")
        print(f"  {rec}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ERP Procurement RAGAS Evaluation Runner")
    parser.add_argument(
        "--routes", nargs="+",
        choices=["sql", "rag", "hybrid"],
        help="Evaluate specific routes only (default: all)"
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run quick 3-question smoke test only (no RAGAS)"
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Don't save results to disk"
    )
    parser.add_argument(
        "--output", default="data/evaluation_results.json",
        help="Path to save results JSON"
    )
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("ERP PROCUREMENT INTELLIGENCE — EVALUATION")
    print("=" * 55)

    # Smoke test only
    if args.smoke_test:
        success = run_smoke_test()
        sys.exit(0 if success else 1)

    # Full evaluation
    from src.evaluation.ragas_eval import run_evaluation

    print(f"\nRunning RAGAS evaluation...")
    if args.routes:
        print(f"  Routes: {args.routes}")
    else:
        print(f"  Routes: all (sql, rag, hybrid)")

    t0      = time.time()
    results = run_evaluation(
        routes_to_eval = args.routes,
        save_results   = not args.no_save,
        output_path    = args.output,
    )
    elapsed = round(time.time() - t0, 1)

    # Detailed output
    print_sample_table(results.samples)
    print_metric_bars(results)
    print_recommendations(results)

    print(f"\nTotal evaluation time: {elapsed}s")
    if not args.no_save:
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
