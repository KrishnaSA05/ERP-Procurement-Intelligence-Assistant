"""
src/evaluation/ragas_eval.py
─────────────────────────────
RAGAS evaluation suite for the procurement RAG pipeline.

Metrics computed:
  • Faithfulness      — is the answer grounded in the retrieved contexts?
  • Answer Relevancy  — does the answer address the question asked?
  • Context Precision — are the retrieved chunks actually relevant?
  • Context Recall    — do the chunks contain what's needed to answer?

How it works:
  1. Load test questions from data/sample_queries.json
  2. Run each RAG question through the full pipeline
  3. Collect (question, answer, contexts, ground_truth) tuples
  4. Feed into RAGAS → get per-metric scores
  5. Save results to data/evaluation_results.json

Usage:
    from src.evaluation.ragas_eval import run_evaluation
    results = run_evaluation()
    print(results.summary())
"""

import os
import sys
import json
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from loguru import logger

# RAGAS imports
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

# Project imports
from src.agents.rag_agent        import RAGAgent
from src.agents.sql_agent        import SQLAgent
from src.graph.langgraph_workflow import build_workflow, run_query
from src.vectorstore.chroma_store import ChromaStore


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class EvalSample:
    """One evaluation sample — maps to one RAGAS row."""
    question     : str
    answer       : str
    contexts     : list[str]      # retrieved chunk texts
    ground_truth : str
    route        : str            # "sql" | "rag" | "hybrid"
    latency_ms   : float = 0.0
    success      : bool  = True
    error        : str   = ""


@dataclass
class EvalResults:
    """Aggregated RAGAS evaluation results."""
    faithfulness       : float
    answer_relevancy   : float
    context_precision  : float
    context_recall     : float
    n_samples          : int
    n_rag_samples      : int      # RAGAS only meaningful for RAG/hybrid
    timestamp          : str
    avg_latency_ms     : float
    samples            : list[EvalSample] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "",
            "=" * 55,
            "RAGAS EVALUATION RESULTS",
            "=" * 55,
            f"  Samples evaluated  : {self.n_samples} total / {self.n_rag_samples} RAG",
            f"  Avg latency        : {self.avg_latency_ms:.0f}ms",
            f"  Timestamp          : {self.timestamp}",
            "",
            f"  Faithfulness       : {self.faithfulness:.4f}",
            f"  Answer Relevancy   : {self.answer_relevancy:.4f}",
            f"  Context Precision  : {self.context_precision:.4f}",
            f"  Context Recall     : {self.context_recall:.4f}",
            "=" * 55,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "metrics": {
                "faithfulness"     : self.faithfulness,
                "answer_relevancy" : self.answer_relevancy,
                "context_precision": self.context_precision,
                "context_recall"   : self.context_recall,
            },
            "meta": {
                "n_samples"      : self.n_samples,
                "n_rag_samples"  : self.n_rag_samples,
                "avg_latency_ms" : self.avg_latency_ms,
                "timestamp"      : self.timestamp,
            },
            "samples": [asdict(s) for s in self.samples],
        }


# ── Test dataset loader ───────────────────────────────────────────────────────

def load_test_questions(path: str = "data/sample_queries.json") -> list[dict]:
    """Load test Q&A pairs from the sample queries file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Test dataset not found: {path}")
    with open(path) as f:
        questions = json.load(f)
    logger.info(f"Loaded {len(questions)} test questions from {path}")
    return questions


# ── Pipeline runner ───────────────────────────────────────────────────────────

def collect_eval_samples(
    test_questions : list[dict],
    workflow       = None,
    rag_agent      : RAGAgent = None,
    routes_to_eval : list[str] = None,   # None = all routes
) -> list[EvalSample]:
    """
    Run each test question through the pipeline and collect
    (question, answer, contexts, ground_truth) tuples.

    For RAGAS context metrics we need the raw retrieved chunks —
    so for RAG/hybrid questions we also run the RAGAgent directly
    to capture the contexts (the workflow only returns the final answer).

    Args:
        test_questions : List of question dicts from sample_queries.json
        workflow       : Compiled LangGraph workflow
        rag_agent      : RAGAgent instance for context capture
        routes_to_eval : Filter to specific routes, e.g. ["rag", "hybrid"]

    Returns:
        List of EvalSample objects
    """
    samples = []

    for i, q in enumerate(test_questions):
        route = q.get("route", "rag")

        # Optionally filter by route
        if routes_to_eval and route not in routes_to_eval:
            continue

        question     = q["question"]
        ground_truth = q.get("ground_truth", "")

        logger.info(f"  [{i+1}/{len(test_questions)}] [{route.upper()}] {question[:70]}")

        t0 = time.time()

        try:
            # ── Get answer from full workflow ──────────────────────────────
            if workflow:
                state = run_query(workflow, question)
                fr    = state.get("final_response")
                answer = fr.final_answer if fr else "No answer generated."
            else:
                answer = "Workflow not available."

            # ── Capture contexts for RAG/hybrid (needed for RAGAS metrics) ─
            contexts = []
            if route in ("rag", "hybrid") and rag_agent:
                rag_result = rag_agent.run(question)
                contexts   = [
                    c.text for c in
                    # Access raw chunks from the store via the rag_agent's retrieve
                    rag_agent._retrieve(question)
                ] if hasattr(rag_agent, "_retrieve") else []

                # Fallback: use citation excerpts if direct chunk access fails
                if not contexts and rag_result.citations:
                    contexts = [c.excerpt for c in rag_result.citations]

            latency_ms = round((time.time() - t0) * 1000, 1)

            samples.append(EvalSample(
                question     = question,
                answer       = answer,
                contexts     = contexts if contexts else ["No context retrieved."],
                ground_truth = ground_truth,
                route        = route,
                latency_ms   = latency_ms,
                success      = True,
            ))

        except Exception as e:
            logger.warning(f"    Failed: {e}")
            samples.append(EvalSample(
                question     = question,
                answer       = f"Error: {e}",
                contexts     = ["Error during retrieval."],
                ground_truth = ground_truth,
                route        = route,
                latency_ms   = 0.0,
                success      = False,
                error        = str(e),
            ))

    logger.success(
        f"Collected {len(samples)} samples "
        f"({sum(1 for s in samples if s.success)} successful)"
    )
    return samples


# ── RAGAS evaluation ──────────────────────────────────────────────────────────

def run_ragas(samples: list[EvalSample]) -> dict:
    """
    Run RAGAS evaluation on a list of EvalSamples.

    Only RAG and hybrid samples are meaningful for RAGAS context metrics
    (SQL-only samples have no retrieved contexts).

    Returns:
        Dict with metric scores (faithfulness, answer_relevancy,
        context_precision, context_recall)
    """
    # Filter to samples with contexts (RAG + hybrid)
    rag_samples = [
        s for s in samples
        if s.success and s.route in ("rag", "hybrid")
        and s.contexts and s.contexts[0] != "No context retrieved."
    ]

    if not rag_samples:
        logger.warning("No RAG samples with contexts found — cannot run RAGAS.")
        return {
            "faithfulness"     : 0.0,
            "answer_relevancy" : 0.0,
            "context_precision": 0.0,
            "context_recall"   : 0.0,
        }

    logger.info(f"Running RAGAS on {len(rag_samples)} samples...")

    # Build HuggingFace Dataset (RAGAS format)
    data = {
        "question"    : [s.question     for s in rag_samples],
        "answer"      : [s.answer       for s in rag_samples],
        "contexts"    : [s.contexts     for s in rag_samples],
        "ground_truth": [s.ground_truth for s in rag_samples],
    }
    dataset = Dataset.from_dict(data)

    # Run evaluation
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    scores = {
        "faithfulness"     : round(float(result["faithfulness"]),      4),
        "answer_relevancy" : round(float(result["answer_relevancy"]),  4),
        "context_precision": round(float(result["context_precision"]), 4),
        "context_recall"   : round(float(result["context_recall"]),    4),
    }

    logger.success(f"RAGAS scores: {scores}")
    return scores


# ── Main evaluation runner ────────────────────────────────────────────────────

def run_evaluation(
    test_path      : str        = "data/sample_queries.json",
    routes_to_eval : list[str]  = None,
    save_results   : bool       = True,
    output_path    : str        = "data/evaluation_results.json",
) -> EvalResults:
    """
    Full evaluation pipeline:
      1. Load test questions
      2. Run pipeline and collect (question, answer, contexts, ground_truth)
      3. Run RAGAS evaluation
      4. Save and return results

    Args:
        test_path      : Path to sample_queries.json
        routes_to_eval : Filter by route type. None = evaluate all.
        save_results   : Write results JSON to disk
        output_path    : Where to save results

    Returns:
        EvalResults with all metric scores
    """
    logger.info("Initialising evaluation pipeline...")

    # Build shared resources
    store     = ChromaStore()
    rag_agent = RAGAgent(store=store)
    workflow  = build_workflow(
        rag_agent   = rag_agent,
        chroma_store= store,
    )

    # Load test questions
    test_questions = load_test_questions(test_path)

    # Collect samples
    logger.info("Collecting evaluation samples...")
    samples = collect_eval_samples(
        test_questions = test_questions,
        workflow       = workflow,
        rag_agent      = rag_agent,
        routes_to_eval = routes_to_eval,
    )

    # Run RAGAS
    scores = run_ragas(samples)

    # Compute latency stats
    successful   = [s for s in samples if s.success]
    avg_latency  = (
        sum(s.latency_ms for s in successful) / len(successful)
        if successful else 0.0
    )
    rag_samples  = [s for s in samples if s.route in ("rag", "hybrid")]

    results = EvalResults(
        faithfulness       = scores["faithfulness"],
        answer_relevancy   = scores["answer_relevancy"],
        context_precision  = scores["context_precision"],
        context_recall     = scores["context_recall"],
        n_samples          = len(samples),
        n_rag_samples      = len(rag_samples),
        avg_latency_ms     = round(avg_latency, 1),
        timestamp          = datetime.now(timezone.utc).isoformat(),
        samples            = samples,
    )

    # Save results
    if save_results:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results.to_dict(), f, indent=2)
        logger.success(f"Results saved to {output_path}")

    print(results.summary())
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation")
    parser.add_argument("--routes", nargs="+",
                        choices=["sql", "rag", "hybrid"],
                        help="Evaluate specific routes only")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save results to disk")
    args = parser.parse_args()

    run_evaluation(
        routes_to_eval = args.routes,
        save_results   = not args.no_save,
    )
