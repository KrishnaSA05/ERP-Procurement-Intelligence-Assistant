# ERP & Procurement Intelligence Assistant

> **Agentic RAG system** that answers cross-source procurement questions by routing between a **SQL agent** (structured ERP data) and a **RAG agent** (vendor contracts + policy documents), synthesising both into a single cited business response.

Built as an AI/ML portfolio project targeting enterprise consulting roles (Capgemini, Infosys, Accenture, Deloitte).

---

## What It Does

Ask it a question in plain English. It figures out where to look, queries the right sources, and returns a cited business answer — in under 2 seconds for SQL queries, ~15 seconds for document retrieval.

| Question | What happens |
|---|---|
| *"Which vendor category has the highest average PO value?"* | Routes to SQL → queries PostgreSQL → narrates result |
| *"What are the payment terms in the Alpha Tech contract?"* | Routes to RAG → retrieves contract chunks → cites sources |
| *"Summarize our IT Services spending and any relevant contract obligations"* | Routes to Hybrid → runs both → Synthesis Agent merges into one answer |

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LangGraph Orchestrator                        │
│                                                                  │
│  ┌──────────────────┐                                            │
│  │  Query Classifier │  ── fast-path rule engine (regex)         │
│  │  (Agent 1)        │  ── LLM fallback for ambiguous queries    │
│  └────────┬─────────┘                                            │
│           │  route: sql | rag | hybrid                           │
│      ┌────┴────────┐                                             │
│      ▼             ▼                                             │
│  ┌─────────┐  ┌─────────┐                                        │
│  │ SQL     │  │ RAG     │                                        │
│  │ Agent   │  │ Agent   │                                        │
│  │(Agent2a)│  │(Agent2b)│                                        │
│  └────┬────┘  └────┬────┘                                        │
│       │            │                                             │
│       ▼            ▼                                             │
│  PostgreSQL    ChromaDB                                          │
│  (vendors,     (vendor contracts                                 │
│   POs,          + procurement                                    │
│   invoices,     policy PDFs)                                     │
│   spend)                                                         │
│       │            │                                             │
│       └─────┬───────┘                                            │
│             ▼                                                    │
│  ┌──────────────────┐                                            │
│  │ Synthesis Agent  │  merges SQL narrative + RAG citations      │
│  │ (Agent 3)        │  into a single business response           │
│  └──────────────────┘                                            │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
  FastAPI /query endpoint
         │
         ▼
  Streamlit Chat UI
```

---

## Tech Stack

| Layer | Local (Dev) | Production (AWS) |
|---|---|---|
| **LLM** | Groq — `llama-3.1-8b-instant` | Amazon Bedrock — Claude Haiku |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | Amazon Bedrock Titan Embed v2 |
| **Orchestration** | LangGraph 0.2.28 | LangGraph 0.2.28 |
| **Vector Store** | ChromaDB 0.5.5 (Docker) | ChromaDB 0.5.5 (Docker / EC2) |
| **Database** | PostgreSQL 16 (Docker) | AWS RDS PostgreSQL |
| **Document Storage** | Local `data/` folder | AWS S3 |
| **API** | FastAPI + Uvicorn | FastAPI on EC2 |
| **UI** | Streamlit | Streamlit on EC2 |
| **Ingestion trigger** | Manual CLI | AWS Lambda (S3 event) |
| **Evaluation** | RAGAS | RAGAS |

> Environment is switched automatically via `APP_ENV=development|production` in `.env` — no code changes needed between local and AWS.

---

## Key Engineering Decisions

**1. Dual-backend LLM client** — `bedrock_client.py` auto-selects Groq (free, fast) for local development and Bedrock (production-grade) for AWS. This eliminates AWS costs during development while keeping production behaviour identical.

**2. Two-tier query classifier** — A regex rule engine handles obvious queries instantly (no LLM call). Only ambiguous queries fall through to the LLM classifier. This reduces latency and API cost on every request.

**3. Safe SQL generation** — The SQL agent uses a two-step pipeline: generate SQL → execute → if error, retry once with error context. `ROUND()` calls always include `::numeric` cast (PostgreSQL float compatibility). Only `SELECT` statements are permitted.

**4. Hybrid synthesis** — The Synthesis Agent receives outputs from both agents and genuinely merges them — it cross-references vendors found in the SQL results against clauses found in the documents, producing a single coherent business answer rather than two separate responses concatenated.

**5. Environment-aware embeddings** — `sentence-transformers` locally (zero cost, 384d), Bedrock Titan in production (1536d). ChromaDB collections are separate per environment to avoid dimension mismatch.

---

## RAGAS Evaluation Results

| Metric | Score |
|---|---|
| Faithfulness | **0.89** |
| Answer Relevancy | **0.92** |
| Context Precision | **0.85** |

Evaluated on 20 procurement questions spanning SQL, RAG, and Hybrid routes. See `notebooks/03_ragas_evaluation.ipynb`.

---

## Quick Start (Local — Groq)

### Prerequisites
- Python 3.11+
- Docker Desktop running
- Groq API key — free at [console.groq.com](https://console.groq.com)

### 1. Clone and install

```bash
git clone https://github.com/yourusername/erp-procurement-intelligence
cd erp-procurement-intelligence

conda create -n mlenv python=3.11
conda activate mlenv

pip install -r requirements.txt
pip install langchain-groq==0.1.9
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` — set these two keys for local development:

```env
APP_ENV=development
GROQ_API_KEY=your_groq_api_key_here
```

Everything else (Postgres, Chroma host/port) can stay as defaults.

### 3. Start infrastructure

```bash
docker-compose up -d

# Verify both containers are healthy (wait ~15s)
docker-compose ps
# erp_postgres   → (healthy)
# erp_chromadb   → (healthy)
```

> Note: `docker-compose.yml` pins ChromaDB to `0.5.5` to match the Python client version. Using `latest` causes a tenant API mismatch.

### 4. Seed the database

```bash
python data/synthetic/generate_erp_data.py
```

Expected:
```
✓ 200 vendors inserted
✓ 500 purchase orders inserted
✓ ~489 invoices inserted
✓ ~154 spend_analysis rows inserted
```

### 5. Ingest contract & policy PDFs

```bash
python src/ingestion/ingest_pipeline.py --all
```

Expected:
```
✓ 3 contracts → 13 chunks → 13 embeddings (vendor_contracts)
✓ 1 policy    →  5 chunks →  5 embeddings (procurement_policies)
```

### 6. Run the application

```bash
# Terminal 1 — FastAPI backend
uvicorn api.main:app --reload --port 8000

# Terminal 2 — Streamlit UI
streamlit run app/streamlit_app.py
```

Open **http://localhost:8501**

Verify system health at **http://localhost:8000/health**

---

## Quick Start (AWS — Bedrock)

### Prerequisites
- AWS account with Bedrock access (Claude Haiku enabled in `us-east-1`)
- Bedrock model access enabled for `anthropic.claude-3-haiku-20240307-v1:0` and `amazon.titan-embed-text-v2:0`

### 1. Set environment to production

```env
APP_ENV=production
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=us-east-1
```

### 2. Provision infrastructure

```bash
python aws/rds_setup.py          # create RDS PostgreSQL instance
python aws/s3_upload.py          # upload PDFs to S3
```

### 3. Run — same commands as local

```bash
uvicorn api.main:app --reload --port 8000
streamlit run app/streamlit_app.py
```

See `infrastructure/deployment_guide.md` for full EC2 setup.

---

## Sample Questions to Try

### SQL Agent (structured ERP data)
```
Which vendor category has the highest average PO value?
Show all invoices overdue by more than 30 days grouped by category.
Who are our top 5 vendors by total PO value?
How many active vendors do we have per country?
What is our total open PO value for IT Services this quarter?
```

### RAG Agent (contracts + policies)
```
What are the payment terms in the Alpha Tech contract?
Does the Beta Logistics contract include a late delivery penalty clause?
What is our policy for single-source procurement above $100K?
Is there a force majeure clause in any of our contracts?
What happens if a vendor terminates the contract early?
```

### Hybrid (both sources combined)
```
Summarize our IT Services spending and any relevant contract obligations.
Which vendors have open POs above $50K and no penalty clause in their contract?
How much do we spend on logistics and what are the contract payment terms?
Which vendors have overdue invoices and what do contracts say about late payment?
```

---

## Project Structure

```
erp-procurement-intelligence/
├── data/
│   ├── synthetic/
│   │   └── generate_erp_data.py     # Faker-based ERP seed data
│   ├── contracts/                   # vendor contract PDFs
│   ├── policies/                    # procurement policy PDFs
│   └── sample_queries.json          # RAGAS evaluation test set
│
├── src/
│   ├── agents/
│   │   ├── bedrock_client.py        # LLM factory — Groq (dev) / Bedrock (prod)
│   │   ├── query_classifier.py      # regex fast-path + LLM fallback router
│   │   ├── sql_agent.py             # NL → SQL → execute → narrate
│   │   ├── rag_agent.py             # embed → retrieve → rerank → answer
│   │   └── synthesis_agent.py       # merge SQL + RAG into one response
│   ├── data/
│   │   ├── schema.py                # SQLAlchemy ORM models
│   │   └── db_loader.py             # connection factory + health check
│   ├── ingestion/
│   │   ├── pdf_loader.py            # PDF → Document objects
│   │   ├── chunker.py               # recursive text splitter
│   │   ├── embedder.py              # MiniLM (dev) / Titan (prod)
│   │   └── ingest_pipeline.py       # orchestrates load→chunk→embed→upsert
│   ├── vectorstore/
│   │   └── chroma_store.py          # ChromaDB client wrapper
│   ├── graph/
│   │   ├── langgraph_workflow.py    # LangGraph DAG definition
│   │   ├── nodes.py                 # one function per graph node
│   │   └── state.py                 # TypedDict workflow state
│   └── evaluation/
│       ├── ragas_eval.py            # RAGAS metrics runner
│       └── eval_runner.py           # batch evaluation harness
│
├── api/
│   ├── main.py                      # FastAPI app — /query /health /history
│   └── models.py                    # Pydantic request/response schemas
│
├── app/
│   └── streamlit_app.py             # Chat UI with route badges + citations
│
├── aws/
│   ├── ec2_setup.sh                 # EC2 bootstrap script
│   ├── rds_setup.py                 # RDS provisioning
│   ├── s3_upload.py                 # PDF upload to S3
│   └── lambda_handler.py            # S3-triggered ingestion Lambda
│
├── notebooks/
│   ├── 01_erp_data_exploration.ipynb
│   ├── 02_contract_chunking_comparison.ipynb
│   └── 03_ragas_evaluation.ipynb
│
├── infrastructure/
│   └── deployment_guide.md
│
├── docker-compose.yml               # PostgreSQL 16 + ChromaDB 0.5.5
├── Dockerfile                       # FastAPI image
├── Dockerfile.streamlit             # Streamlit image
├── requirements.txt
└── .env.example
```

---

## Data

| Source | Type | Size | Notes |
|---|---|---|---|
| ERP tables | Synthetic (Faker) | ~1,200 rows | Vendors, POs, invoices, spend — realistic domain logic |
| Vendor contracts | Public-domain PDFs | 3 contracts, 13 chunks | Penalty, payment, termination clauses |
| Procurement policy | Public-domain PDF | 1 document, 5 chunks | Approval thresholds, compliance rules |

Synthetic ERP + real public-domain contract templates mirrors consulting pre-sales practice: realistic data before client data is available under NDA.

---

## License

MIT
