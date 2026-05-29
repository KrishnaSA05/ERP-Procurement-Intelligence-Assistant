"""
aws/s3_upload.py
─────────────────
Uploads contract and policy PDFs to S3.
Creates the bucket if it doesn't exist.

Usage:
    python aws/s3_upload.py --create-bucket    # create S3 bucket
    python aws/s3_upload.py --upload           # upload all PDFs
    python aws/s3_upload.py --list             # list bucket contents
    python aws/s3_upload.py --all              # create bucket + upload
"""

import os
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

AWS_REGION  = os.getenv("AWS_REGION",       "us-east-1")
BUCKET_NAME = os.getenv("S3_BUCKET_NAME",   "erp-procurement-docs")
PREFIX_CON  = os.getenv("S3_PREFIX_CONTRACTS", "contracts/")
PREFIX_POL  = os.getenv("S3_PREFIX_POLICIES",  "policies/")


def get_s3():
    return boto3.client("s3", region_name=AWS_REGION)


# ── Create bucket ─────────────────────────────────────────────────────────────

def create_bucket():
    s3 = get_s3()
    try:
        if AWS_REGION == "us-east-1":
            s3.create_bucket(Bucket=BUCKET_NAME)
        else:
            s3.create_bucket(
                Bucket=BUCKET_NAME,
                CreateBucketConfiguration={"LocationConstraint": AWS_REGION},
            )

        # Block all public access
        s3.put_public_access_block(
            Bucket=BUCKET_NAME,
            PublicAccessBlockConfiguration={
                "BlockPublicAcls"      : True,
                "IgnorePublicAcls"     : True,
                "BlockPublicPolicy"    : True,
                "RestrictPublicBuckets": True,
            },
        )

        # Enable versioning
        s3.put_bucket_versioning(
            Bucket=BUCKET_NAME,
            VersioningConfiguration={"Status": "Enabled"},
        )

        logger.success(f"  ✓ Bucket created: s3://{BUCKET_NAME}")

    except ClientError as e:
        if e.response["Error"]["Code"] == "BucketAlreadyOwnedByYou":
            logger.info(f"  Bucket already exists: s3://{BUCKET_NAME}")
        else:
            raise


# ── Upload PDFs ───────────────────────────────────────────────────────────────

def upload_pdfs():
    s3 = get_s3()

    upload_map = {
        ROOT / "data" / "contracts": PREFIX_CON,
        ROOT / "data" / "policies" : PREFIX_POL,
    }

    total = 0
    for local_dir, s3_prefix in upload_map.items():
        if not local_dir.exists():
            logger.warning(f"  Directory not found: {local_dir} — skipping")
            continue

        pdfs = list(local_dir.glob("*.pdf"))
        if not pdfs:
            logger.warning(f"  No PDFs in {local_dir} — skipping")
            continue

        logger.info(f"Uploading {len(pdfs)} PDFs from {local_dir.name}/...")

        for pdf in pdfs:
            s3_key = s3_prefix + pdf.name
            try:
                s3.upload_file(
                    str(pdf),
                    BUCKET_NAME,
                    s3_key,
                    ExtraArgs={"ContentType": "application/pdf"},
                )
                logger.success(f"  ✓ {pdf.name} → s3://{BUCKET_NAME}/{s3_key}")
                total += 1
            except Exception as e:
                logger.error(f"  ✗ Failed to upload {pdf.name}: {e}")

    logger.success(f"\nUploaded {total} files to s3://{BUCKET_NAME}")


# ── List bucket ───────────────────────────────────────────────────────────────

def list_bucket():
    s3 = get_s3()
    try:
        response = s3.list_objects_v2(Bucket=BUCKET_NAME)
        objects  = response.get("Contents", [])

        if not objects:
            print(f"Bucket s3://{BUCKET_NAME} is empty.")
            return

        print(f"\nContents of s3://{BUCKET_NAME}:")
        for obj in objects:
            size_kb = round(obj["Size"] / 1024, 1)
            print(f"  {obj['Key']:<60} {size_kb:>8} KB")
        print(f"\nTotal: {len(objects)} objects")

    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchBucket":
            logger.error(f"Bucket '{BUCKET_NAME}' does not exist. Run --create-bucket first.")
        else:
            raise


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="S3 PDF upload manager")
    parser.add_argument("--create-bucket", action="store_true")
    parser.add_argument("--upload",        action="store_true")
    parser.add_argument("--list",          action="store_true")
    parser.add_argument("--all",           action="store_true",
                        help="Create bucket + upload all PDFs")
    args = parser.parse_args()

    if args.all or args.create_bucket: create_bucket()
    if args.all or args.upload:        upload_pdfs()
    if args.list:                      list_bucket()
