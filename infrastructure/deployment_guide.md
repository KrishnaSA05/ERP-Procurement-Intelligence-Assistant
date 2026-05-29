# AWS Deployment Guide
## ERP Procurement Intelligence Assistant

Complete step-by-step guide to deploy the full stack on AWS Free Tier.

---

## Architecture on AWS

```
Internet
    │
    ▼
EC2 t2.micro
├── Docker: FastAPI (port 8000)
├── Docker: Streamlit (port 8501)
└── Docker: ChromaDB (port 8001, internal only)
    │
    ├──► RDS PostgreSQL t3.micro (ERP data)
    ├──► S3 Bucket (contract PDFs)
    ├──► Amazon Bedrock (Claude Haiku — LLM)
    └──► Lambda (triggered by S3 upload → runs ingestion)
```

---

## Prerequisites

- AWS account with Free Tier active
- AWS CLI installed and configured (`aws configure`)
- Bedrock access enabled for Claude Haiku in us-east-1
  → AWS Console → Bedrock → Model Access → Enable Claude Haiku

---

## Step 1 — Launch EC2

1. Go to EC2 → Launch Instance
2. Settings:
   - **Name**: erp-procurement-server
   - **AMI**: Ubuntu Server 22.04 LTS (Free Tier eligible)
   - **Instance type**: t2.micro
   - **Key pair**: Create new → download `.pem` file
   - **Security group**: Create new with these inbound rules:

| Type | Port | Source |
|------|------|--------|
| SSH | 22 | My IP |
| Custom TCP | 8000 | My IP (FastAPI) |
| Custom TCP | 8501 | My IP (Streamlit) |

3. Launch instance → note the **Public IPv4 address**

---

## Step 2 — Set up RDS PostgreSQL

```bash
# From your LOCAL machine
python aws/rds_setup.py --create
```

This will:
- Create `db.t3.micro` PostgreSQL instance (Free Tier)
- Wait ~8 minutes for it to become available
- Update your `.env` with the RDS endpoint automatically

> **Security group**: Add an inbound rule on the RDS security group to allow
> port 5432 from the EC2 security group (not 0.0.0.0/0).

---

## Step 3 — Set up S3 + Upload PDFs

```bash
# Create bucket and upload PDFs
python aws/s3_upload.py --all
```

Note the bucket name from your `.env`: `S3_BUCKET_NAME`

---

## Step 4 — Deploy to EC2

```bash
# SSH into EC2
ssh -i your-key.pem ubuntu@YOUR_EC2_PUBLIC_IP

# Run setup script (installs Docker, clones repo, starts services)
curl -fsSL https://raw.githubusercontent.com/yourusername/erp-procurement-intelligence/main/aws/ec2_setup.sh | bash
```

Or manually:
```bash
scp -i your-key.pem aws/ec2_setup.sh ubuntu@YOUR_EC2_IP:~/
ssh -i your-key.pem ubuntu@YOUR_EC2_IP
chmod +x ec2_setup.sh && ./ec2_setup.sh
```

---

## Step 5 — Configure .env on EC2

```bash
ssh -i your-key.pem ubuntu@YOUR_EC2_IP
nano /opt/erp-procurement/.env
```

Update these values:
```env
APP_ENV=production

# RDS (from Step 2 — already set if you ran rds_setup.py)
DATABASE_URL=postgresql://erp_user:your_password@YOUR_RDS_ENDPOINT:5432/erp_procurement

# AWS Bedrock
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret

# S3
S3_BUCKET_NAME=erp-procurement-docs

# ChromaDB (internal Docker network)
CHROMA_HOST=chromadb
CHROMA_PORT=8000
```

---

## Step 6 — Seed Data and Ingest Documents

```bash
# On EC2, inside the project directory
cd /opt/erp-procurement

# Seed ERP data into RDS
python aws/rds_setup.py --seed

# Ingest PDFs from local data/ folder into ChromaDB
python src/ingestion/ingest_pipeline.py --all

# Or ingest from S3 (trigger Lambda manually for testing)
python aws/s3_upload.py --upload
```

---

## Step 7 — Restart Services

```bash
cd /opt/erp-procurement
docker compose -f docker-compose.prod.yml restart
docker compose -f docker-compose.prod.yml ps
```

---

## Step 8 — Set up Lambda (Auto-ingestion on S3 Upload)

1. Go to AWS Lambda → Create Function
2. **Name**: erp-procurement-ingest
3. **Runtime**: Python 3.11
4. Upload `aws/lambda_handler.py` as deployment package
5. Set environment variables:
   - `CHROMA_HOST` → EC2 private IP
   - `CHROMA_PORT` → 8001
   - `APP_ENV` → production
6. Add trigger: S3 → your bucket → Event type: PUT
7. Set Lambda execution role with permissions:
   - `AmazonS3ReadOnlyAccess`
   - `AmazonBedrockFullAccess`

---

## Verify Everything Works

```bash
# Health check
curl http://YOUR_EC2_IP:8000/health

# Test a query
curl -X POST http://YOUR_EC2_IP:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How many vendors do we have?"}'
```

Open in browser:
- **Streamlit UI**: `http://YOUR_EC2_IP:8501`
- **API Docs**:     `http://YOUR_EC2_IP:8000/docs`

---

## Useful Commands on EC2

```bash
# View logs
docker compose -f docker-compose.prod.yml logs -f api
docker compose -f docker-compose.prod.yml logs -f streamlit

# Restart a single service
docker compose -f docker-compose.prod.yml restart api

# Stop everything
docker compose -f docker-compose.prod.yml down

# Rebuild after code change
git pull && docker compose -f docker-compose.prod.yml up -d --build api
```

---

## Monthly Cost Estimate (Free Tier)

| Service | Free Tier | Notes |
|---------|-----------|-------|
| EC2 t2.micro | 750 hrs/month | Enough for 1 instance 24/7 |
| RDS t3.micro | 750 hrs/month | 20GB storage included |
| S3 | 5GB free | PDFs are <100MB |
| Lambda | 1M requests/month | Only fires on new PDF uploads |
| Bedrock Claude Haiku | ~$0.25 per 1M input tokens | Very cheap for demo usage |

**Estimated cost for demo usage: $0–$2/month** after Free Tier.

---

## Teardown (when done)

```bash
# Delete RDS (saves ~$0 but good practice)
python aws/rds_setup.py --delete

# Stop EC2 (or terminate to avoid storage charges)
# AWS Console → EC2 → Stop/Terminate

# Empty and delete S3 bucket
aws s3 rm s3://erp-procurement-docs --recursive
aws s3 rb s3://erp-procurement-docs
```
