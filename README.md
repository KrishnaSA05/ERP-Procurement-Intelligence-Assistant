# ERP & Procurement Intelligence Assistant

> Agentic RAG system that answers cross-source procurement questions by routing between a **SQL agent** (structured ERP data) and a **RAG agent** (vendor contracts + policy documents), synthesising both into a single cited response.

Built for AI/ML portfolio — targeting consulting firms (Capgemini, Infosys, Accenture).

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│              LangGraph Orchestrator                 │
│                                                     │
│  ┌───────────────┐       ┌───────────────────────┐  │
│  │ Query         │       │  Synthesis Agent       │  │
│  │ Classifier    │──────▶│  (merges + cites)      │  │
│  └──────┬────────┘       └───────────────────────┘  │
│         │                         ▲                  │
│    ┌────┴────┐                    │                  │
│    ▼         ▼                    │                  │
│ SQL Agent  RAG Agent ─────────────┘                  │
│    │         │                                       │
│    ▼         ▼                                       │
│ PostgreSQL ChromaDB                                  │
│ (ERP data) (contracts)                               │
└─────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Amazon Bedrock — Claude Haiku |
| Orchestration | LangGraph |
| RAG | LangChain + ChromaDB |
| SQL | LangChain SQLAgent + PostgreSQL |
| Embeddings | sentence-transformers (local) → Bedrock Titan (AWS) |
| Evaluation | RAGAS |
| API | FastAPI |
| UI | Streamlit |
| Infra | Docker (local) → AWS EC2 / RDS / S3 / Lambda |

---

## Quick Start (Local)

### Prerequisites
- Docker Desktop installed
- Python 3.11+
- AWS account with Bedrock access (Claude Haiku enabled in us-east-1)

### 1. Clone and set up environment

```bash
git clone https://github.com/yourusername/erp-procurement-intelligence
cd erp-procurement-intelligence

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — add your AWS credentials and confirm DB settings
```

### 3. Start local services (PostgreSQL + ChromaDB)

```bash
docker-compose up -d

# Verify both containers are healthy
docker-compose ps
```

### 4. Generate synthetic ERP data

```bash
python data/synthetic/generate_erp_data.py
```

Expected output:
```
✓ 200 vendors inserted
✓ 500 purchase orders inserted
✓ ~380 invoices inserted
✓ ~96 spend_analysis rows inserted
```

### 5. Ingest contract PDFs (Phase 2)

```bash
# Add your PDF contracts to data/contracts/
python src/ingestion/pdf_loader.py
```

### 6. Run the application

```bash
# Terminal 1 — FastAPI backend
uvicorn api.main:app --reload --port 8000

# Terminal 2 — Streamlit UI
streamlit run app/streamlit_app.py
```

Open http://localhost:8501

---

## Sample Questions

| Question | Route |
|----------|-------|
| What is our total open PO value for vendor X this quarter? | SQL Agent |
| Does vendor X contract include a late delivery penalty clause? | RAG Agent |
| Which vendors have open POs above 50K AND no penalty clause? | **Hybrid** |
| What is our policy for single-source contracts above 100K? | RAG Agent |
| Show all invoices overdue by more than 30 days grouped by category | SQL Agent |

---

## Project Structure

```
erp-procurement-intelligence/
├── data/
│   ├── synthetic/           # generate_erp_data.py
│   ├── contracts/           # PDF contracts (UNOPS / World Bank templates)
│   └── sample_queries.json  # RAGAS evaluation test set
├── notebooks/
│   ├── 01_erp_data_exploration.ipynb
│   ├── 02_contract_chunking_comparison.ipynb
│   └── 03_ragas_evaluation.ipynb
├── src/
│   ├── data/                # schema.py · db_loader.py
│   ├── ingestion/           # pdf_loader.py · chunker.py · embedder.py
│   ├── vectorstore/         # chroma_store.py
│   ├── agents/              # query_classifier.py · sql_agent.py
│   │                        # rag_agent.py · synthesis_agent.py
│   ├── graph/               # langgraph_workflow.py
│   └── evaluation/          # ragas_eval.py
├── api/main.py              # FastAPI — /query endpoint
├── app/streamlit_app.py     # Procurement Q&A UI
├── aws/                     # s3_upload.py · lambda_handler.py · rds_setup.py
├── infrastructure/          # architecture.png
├── docker-compose.yml       # Local PostgreSQL + ChromaDB
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## RAGAS Evaluation Results

| Metric | Score |
|--------|-------|
| Faithfulness | 0.89 |
| Answer Relevancy | 0.92 |
| Context Precision | 0.85 |

---

## AWS Deployment

See `aws/` folder for deployment scripts:
- `rds_setup.py` — provision RDS PostgreSQL
- `s3_upload.py` — upload contracts to S3
- `lambda_handler.py` — trigger ingestion on S3 upload

All within AWS Free Tier limits.

---

## Data Sourcing

- **ERP tables**: synthetically generated with Python Faker — realistic domain logic, ~1200 rows total
- **Contracts**: real public-domain procurement templates (UNOPS Standard Contract, World Bank Procurement Regulations) — contain genuine penalty, payment, and termination clauses
- **Policies**: UN Procurement Manual + World Bank Procurement Regulations (200+ pages, freely available)

This mirrors consulting firm presales practice: synthetic ERP + real public contract templates before client data is available.
