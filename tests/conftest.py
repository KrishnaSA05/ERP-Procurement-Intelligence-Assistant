"""
Shared pytest fixtures for ERP Intelligence Assistant.
Covers: DB session, test client, mocked Bedrock client, in-memory Chroma.
"""
import os
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


# ── Environment defaults for testing ─────────────────────────────────────────
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
os.environ.setdefault("POSTGRES_DB", "erp_db")
os.environ.setdefault("POSTGRES_USER", "erp_user")
os.environ.setdefault("POSTGRES_PASSWORD", "erp_pass")
os.environ.setdefault("CHROMA_PERSIST_DIR", "/tmp/chroma_test")
os.environ.setdefault("EMBEDDING_PROVIDER", "local")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")


# ── Database ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def db_engine():
    """Create a test database engine (uses env vars, falls back to SQLite)."""
    pg_url = (
        f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
        f"@{os.environ['POSTGRES_HOST']}:{os.environ['POSTGRES_PORT']}"
        f"/{os.environ['POSTGRES_DB']}"
    )
    try:
        engine = create_engine(pg_url, pool_pre_ping=True)
        engine.connect()
        return engine
    except Exception:
        # Fall back to in-memory SQLite for pure unit tests
        return create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})


@pytest.fixture
def db_session(db_engine):
    """Provide a transactional test DB session (rolled back after each test)."""
    Session = sessionmaker(bind=db_engine)
    session = Session()
    yield session
    session.rollback()
    session.close()


# ── Bedrock / LLM mock ────────────────────────────────────────────────────────
@pytest.fixture
def mock_bedrock_client():
    """Return a mock Bedrock client that yields deterministic responses."""
    mock = MagicMock()
    mock.invoke_model.return_value = {
        "body": MagicMock(
            read=lambda: b'{"content": [{"text": "Mocked LLM response for testing."}]}'
        )
    }
    return mock


@pytest.fixture
def mock_llm_chain():
    """Patch the LangChain LLM so no real API calls are made."""
    with patch("src.agents.bedrock_client.get_llm") as mock_llm:
        mock_llm.return_value.invoke = AsyncMock(return_value="Mocked response")
        yield mock_llm


# ── ChromaDB mock ─────────────────────────────────────────────────────────────
@pytest.fixture
def mock_chroma_collection():
    """In-memory Chroma collection for RAG tests."""
    mock = MagicMock()
    mock.query.return_value = {
        "documents": [["Sample contract clause about payment terms."]],
        "metadatas": [[{"source": "contract_001.pdf", "page": 1}]],
        "distances": [[0.15]],
    }
    return mock


# ── FastAPI test client ───────────────────────────────────────────────────────
@pytest_asyncio.fixture
async def api_client():
    """Async HTTP client for FastAPI endpoint tests."""
    try:
        from api.main import app
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver"
        ) as client:
            yield client
    except ImportError:
        pytest.skip("api.main not importable — skipping API tests")


# ── Sample data helpers ───────────────────────────────────────────────────────
@pytest.fixture
def sample_query_structured():
    return {
        "query": "What is the total spend for vendor Acme Corp in Q1 2024?",
        "session_id": "test-session-001",
    }


@pytest.fixture
def sample_query_unstructured():
    return {
        "query": "What are the payment terms in the master service agreement?",
        "session_id": "test-session-002",
    }


@pytest.fixture
def sample_query_hybrid():
    return {
        "query": "Compare the contract value for Acme Corp with their actual invoiced amount.",
        "session_id": "test-session-003",
    }
