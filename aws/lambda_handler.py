"""
aws/lambda_handler.py
──────────────────────
AWS Lambda function triggered by S3 PUT events.

When a PDF is uploaded to s3://{bucket}/contracts/ or s3://{bucket}/policies/,
this Lambda downloads it and triggers the ingestion pipeline
(chunk → embed → upsert to ChromaDB on EC2).

Deployment:
  1. Zip this file + dependencies
  2. Upload to Lambda
  3. Set trigger: S3 PUT event on your bucket
  4. Set env vars: CHROMA_HOST, CHROMA_PORT, EMBED_MODEL, APP_ENV

Environment variables (set in Lambda console):
  CHROMA_HOST  — EC2 private IP running ChromaDB
  CHROMA_PORT  — 8001
  APP_ENV      — production
  AWS_REGION   — us-east-1
"""

import os
import json
import tempfile
import urllib.parse
import boto3
from loguru import logger


def handler(event, context):
    """
    Lambda entry point.
    Triggered by S3 ObjectCreated events.
    """
    logger.info(f"Lambda triggered: {json.dumps(event, indent=2)}")

    s3_client = boto3.client("s3")
    processed = []
    errors    = []

    for record in event.get("Records", []):
        # Only process S3 PUT events
        if record.get("eventName", "").startswith("ObjectCreated"):
            bucket = record["s3"]["bucket"]["name"]
            key    = urllib.parse.unquote_plus(record["s3"]["object"]["key"])

            # Only process PDFs
            if not key.lower().endswith(".pdf"):
                logger.info(f"Skipping non-PDF: {key}")
                continue

            logger.info(f"Processing: s3://{bucket}/{key}")

            try:
                result = process_pdf(s3_client, bucket, key)
                processed.append({"key": key, "chunks": result})
                logger.success(f"  ✓ {key} → {result} chunks ingested")
            except Exception as e:
                logger.error(f"  ✗ Failed: {key} — {e}")
                errors.append({"key": key, "error": str(e)})

    return {
        "statusCode": 200 if not errors else 207,
        "body": json.dumps({
            "processed": processed,
            "errors"   : errors,
            "total"    : len(processed),
        }),
    }


def process_pdf(s3_client, bucket: str, key: str) -> int:
    """
    Download PDF from S3, run ingestion pipeline, return chunk count.

    Args:
        s3_client : boto3 S3 client
        bucket    : S3 bucket name
        key       : S3 object key (e.g. "contracts/vendor_contract_alpha.pdf")

    Returns:
        Number of chunks ingested into ChromaDB
    """
    # Infer doc_type from S3 prefix
    doc_type = "contract" if key.startswith("contracts/") else "policy"
    filename = key.split("/")[-1]

    # Download PDF to Lambda /tmp (max 512MB)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = tmp.name

    s3_client.download_file(bucket, key, tmp_path)
    logger.info(f"  Downloaded to {tmp_path}")

    # Run ingestion pipeline
    # These imports work when Lambda layer includes the project deps
    from src.ingestion.pdf_loader    import load_pdf
    from src.ingestion.chunker       import chunk_documents
    from src.ingestion.embedder      import get_embedder, embed_chunks
    from src.vectorstore.chroma_store import ChromaStore

    doc      = load_pdf(tmp_path)
    doc.doc_type    = doc_type
    doc.source_file = filename
    doc.doc_id      = filename.replace(".pdf", "").lower().replace(" ", "_")

    chunks   = chunk_documents([doc])
    embedder = get_embedder()             # uses Bedrock Titan in production
    embedded = embed_chunks(chunks, embedder)

    store  = ChromaStore(
        host = os.getenv("CHROMA_HOST", "localhost"),
        port = int(os.getenv("CHROMA_PORT", 8001)),
    )
    counts = store.upsert(embedded)

    # Cleanup
    os.unlink(tmp_path)

    return len(embedded)
