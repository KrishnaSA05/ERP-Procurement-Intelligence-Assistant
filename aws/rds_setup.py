"""
aws/rds_setup.py
─────────────────
Provisions an RDS PostgreSQL instance and populates it with ERP data.

Steps:
  1. Create RDS instance (t3.micro — Free Tier)
  2. Wait for it to become available
  3. Update .env with the RDS endpoint
  4. Run SQLAlchemy schema creation
  5. Run the ERP data generator against RDS

Run:
    python aws/rds_setup.py --create     # provision new RDS instance
    python aws/rds_setup.py --status     # check existing instance
    python aws/rds_setup.py --seed       # seed data into existing RDS
    python aws/rds_setup.py --delete     # tear down (CAUTION)
"""

import os
import sys
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import boto3
from dotenv import load_dotenv, set_key
from loguru import logger

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

AWS_REGION        = os.getenv("AWS_REGION", "us-east-1")
RDS_INSTANCE_ID   = "erp-procurement-db"
RDS_ENGINE        = "postgres"
RDS_ENGINE_VER    = "16.1"
RDS_INSTANCE_CLASS= "db.t3.micro"          # Free Tier eligible
RDS_STORAGE_GB    = 20
RDS_DB_NAME       = os.getenv("POSTGRES_DB",       "erp_procurement")
RDS_USERNAME      = os.getenv("POSTGRES_USER",     "erp_user")
RDS_PASSWORD      = os.getenv("POSTGRES_PASSWORD", "erp_pass_CHANGE_ME")
RDS_PORT          = 5432

ENV_FILE = ROOT / ".env"


# ── RDS Client ────────────────────────────────────────────────────────────────

def get_rds_client():
    return boto3.client("rds", region_name=AWS_REGION)


# ── Create RDS instance ───────────────────────────────────────────────────────

def create_rds_instance():
    """Provision a new RDS PostgreSQL instance."""
    client = get_rds_client()

    logger.info(f"Creating RDS instance '{RDS_INSTANCE_ID}'...")
    logger.info(f"  Engine  : {RDS_ENGINE} {RDS_ENGINE_VER}")
    logger.info(f"  Class   : {RDS_INSTANCE_CLASS} (Free Tier)")
    logger.info(f"  Storage : {RDS_STORAGE_GB}GB")
    logger.warning("  ⚠ Make sure your EC2 security group allows port 5432 from EC2")

    try:
        response = client.create_db_instance(
            DBInstanceIdentifier  = RDS_INSTANCE_ID,
            DBInstanceClass       = RDS_INSTANCE_CLASS,
            Engine                = RDS_ENGINE,
            EngineVersion         = RDS_ENGINE_VER,
            MasterUsername        = RDS_USERNAME,
            MasterUserPassword    = RDS_PASSWORD,
            DBName                = RDS_DB_NAME,
            AllocatedStorage      = RDS_STORAGE_GB,
            StorageType           = "gp2",
            PubliclyAccessible    = False,   # only accessible from within VPC
            MultiAZ               = False,   # single-AZ for Free Tier
            BackupRetentionPeriod = 1,
            Tags = [
                {"Key": "Project", "Value": "erp-procurement-intelligence"},
                {"Key": "Environment", "Value": "production"},
            ],
        )

        instance = response["DBInstance"]
        logger.success(f"RDS instance creation initiated: {instance['DBInstanceIdentifier']}")
        logger.info("Waiting for instance to become available (5–10 mins)...")

        # Poll until available
        waiter = client.get_waiter("db_instance_available")
        waiter.wait(
            DBInstanceIdentifier = RDS_INSTANCE_ID,
            WaiterConfig={"Delay": 30, "MaxAttempts": 40},
        )

        # Get endpoint
        endpoint = _get_endpoint()
        logger.success(f"  ✓ RDS ready at: {endpoint}")

        # Update .env
        _update_env(endpoint)
        return endpoint

    except client.exceptions.DBInstanceAlreadyExistsFault:
        logger.warning(f"Instance '{RDS_INSTANCE_ID}' already exists.")
        endpoint = _get_endpoint()
        logger.info(f"  Existing endpoint: {endpoint}")
        _update_env(endpoint)
        return endpoint


def _get_endpoint() -> str:
    """Get the RDS endpoint hostname."""
    client   = get_rds_client()
    response = client.describe_db_instances(DBInstanceIdentifier=RDS_INSTANCE_ID)
    instance = response["DBInstances"][0]
    return instance["Endpoint"]["Address"]


def _update_env(endpoint: str):
    """Update DATABASE_URL in .env with the RDS endpoint."""
    db_url = (
        f"postgresql://{RDS_USERNAME}:{RDS_PASSWORD}"
        f"@{endpoint}:{RDS_PORT}/{RDS_DB_NAME}"
    )
    set_key(str(ENV_FILE), "POSTGRES_HOST", endpoint)
    set_key(str(ENV_FILE), "DATABASE_URL",  db_url)
    logger.success(f"  .env updated with RDS endpoint")
    logger.info(f"  DATABASE_URL={db_url.split('@')[0]}@***")   # mask creds in log


# ── Check status ──────────────────────────────────────────────────────────────

def check_status():
    """Print current RDS instance status."""
    client = get_rds_client()
    try:
        response = client.describe_db_instances(DBInstanceIdentifier=RDS_INSTANCE_ID)
        instance = response["DBInstances"][0]
        print(f"\nRDS Instance: {instance['DBInstanceIdentifier']}")
        print(f"  Status    : {instance['DBInstanceStatus']}")
        print(f"  Class     : {instance['DBInstanceClass']}")
        print(f"  Engine    : {instance['Engine']} {instance['EngineVersion']}")
        if "Endpoint" in instance:
            print(f"  Endpoint  : {instance['Endpoint']['Address']}")
        print(f"  Storage   : {instance['AllocatedStorage']}GB")
    except client.exceptions.DBInstanceNotFoundFault:
        print(f"Instance '{RDS_INSTANCE_ID}' not found.")


# ── Seed data ─────────────────────────────────────────────────────────────────

def seed_data():
    """Run schema creation and data generator against RDS."""
    load_dotenv(override=True)   # reload to pick up updated DATABASE_URL

    logger.info("Creating schema on RDS...")
    from src.data.schema   import Base
    from src.data.db_loader import get_engine

    engine = get_engine()
    Base.metadata.create_all(engine)
    logger.success("  ✓ Tables created")

    logger.info("Seeding ERP data...")
    # Import and run the generator
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "generate_erp_data",
        ROOT / "data" / "synthetic" / "generate_erp_data.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


# ── Delete instance ───────────────────────────────────────────────────────────

def delete_instance():
    """Delete the RDS instance. CAUTION: this is irreversible."""
    confirm = input(
        f"⚠ CAUTION: This will DELETE '{RDS_INSTANCE_ID}' and all data.\n"
        f"Type the instance ID to confirm: "
    )
    if confirm != RDS_INSTANCE_ID:
        print("Aborted.")
        return

    client = get_rds_client()
    client.delete_db_instance(
        DBInstanceIdentifier      = RDS_INSTANCE_ID,
        SkipFinalSnapshot         = True,
        DeleteAutomatedBackups    = True,
    )
    logger.warning(f"Deletion initiated for '{RDS_INSTANCE_ID}'")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RDS PostgreSQL provisioning")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--create", action="store_true", help="Create RDS instance")
    group.add_argument("--status", action="store_true", help="Check instance status")
    group.add_argument("--seed",   action="store_true", help="Seed ERP data into RDS")
    group.add_argument("--delete", action="store_true", help="Delete RDS instance")
    args = parser.parse_args()

    if args.create : create_rds_instance()
    if args.status : check_status()
    if args.seed   : seed_data()
    if args.delete : delete_instance()
