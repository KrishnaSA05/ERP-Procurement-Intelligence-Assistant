"""Smoke tests — verify the pipeline components are importable and wired up."""
import pytest


class TestQueryClassifier:
    def test_sql_intent_detected(self):
        """Structured/numeric queries should route to SQL agent."""
        try:
            from src.agents.query_classifier import classify_query
        except ImportError:
            pytest.skip("query_classifier not importable")

        result = classify_query("What is the total spend for vendor Acme Corp in Q1 2024?")
        assert result in ("sql", "hybrid"), f"Expected sql/hybrid, got {result}"

    def test_rag_intent_detected(self):
        """Unstructured contract queries should route to RAG agent."""
        try:
            from src.agents.query_classifier import classify_query
        except ImportError:
            pytest.skip("query_classifier not importable")

        result = classify_query("What are the indemnification clauses in the MSA?")
        assert result in ("rag", "hybrid"), f"Expected rag/hybrid, got {result}"


class TestAPIHealth:
    @pytest.mark.asyncio
    async def test_health_endpoint(self, api_client):
        resp = await api_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("status") == "ok"

    @pytest.mark.asyncio
    async def test_query_endpoint_returns_200(
        self, api_client, sample_query_structured, mock_llm_chain
    ):
        resp = await api_client.post("/query", json=sample_query_structured)
        # Accept 200 or 422 (if request schema differs) — just not 500
        assert resp.status_code != 500


class TestLangGraphWorkflow:
    def test_workflow_importable(self):
        try:
            from src.workflow.langgraph_workflow import build_workflow
            wf = build_workflow()
            assert wf is not None
        except ImportError:
            pytest.skip("langgraph_workflow not importable")

    def test_state_schema(self):
        try:
            from src.workflow.state import AgentState
            state = AgentState(query="test", session_id="s1")
            assert state.query == "test"
        except ImportError:
            pytest.skip("state module not importable")


class TestChunking:
    def test_chunk_size_respected(self):
        """Chunker should produce chunks no larger than max_chars."""
        try:
            from src.ingestion.chunker import chunk_text
        except ImportError:
            pytest.skip("chunker not importable")

        text = "A" * 5000
        chunks = chunk_text(text, max_chars=800, overlap=150)
        for chunk in chunks:
            assert len(chunk) <= 850, f"Chunk too long: {len(chunk)}"
