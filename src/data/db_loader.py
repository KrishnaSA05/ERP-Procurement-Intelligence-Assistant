"""
src/data/db_loader.py
──────────────────────
Database connection factory and utility helpers.
Used by SQL agent, data generator, and API.
"""

import os
from functools import lru_cache
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from loguru import logger

load_dotenv()


# ── Engine factory ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_engine():
    """
    Returns a cached SQLAlchemy engine.
    lru_cache ensures a single engine is reused across the app.
    """
    url = os.getenv(
        "DATABASE_URL",
        "postgresql://erp_user:erp_pass@localhost:5432/erp_procurement"
    )
    logger.info(f"Creating DB engine → {url.split('@')[-1]}")   # log host only, not creds
    return create_engine(
        url,
        pool_pre_ping=True,      # verify connections before use
        pool_size=5,
        max_overflow=10,
        echo=False,
    )


def get_session() -> Session:
    """Returns a new SQLAlchemy session. Caller is responsible for closing."""
    engine  = get_engine()
    Factory = sessionmaker(bind=engine)
    return Factory()


# ── Connection string (used by LangChain SQLDatabaseChain) ────────────────────

def get_db_uri() -> str:
    return os.getenv(
        "DATABASE_URL",
        "postgresql://erp_user:erp_pass@localhost:5432/erp_procurement"
    )


# ── Health check ──────────────────────────────────────────────────────────────

def check_connection() -> dict:
    """
    Verify the DB is reachable and return table row counts.
    Useful for /health endpoint and startup checks.
    """
    try:
        engine = get_engine()
        with engine.connect() as conn:
            tables = ["vendors", "purchase_orders", "invoices", "spend_analysis"]
            counts = {}
            for table in tables:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                counts[table] = result.scalar()
        logger.success("DB health check passed.")
        return {"status": "ok", "row_counts": counts}
    except Exception as e:
        logger.error(f"DB health check failed: {e}")
        return {"status": "error", "detail": str(e)}


# ── Schema description (injected into SQL agent prompt) ───────────────────────

SCHEMA_DESCRIPTION = """
You have access to a PostgreSQL database with the following tables:

TABLE: vendors
  vendor_id      INT PRIMARY KEY
  name           VARCHAR   -- company name of the supplier
  country        VARCHAR   -- country of the vendor
  category       VARCHAR   -- one of: IT Services, Logistics, Consulting, Facilities,
                              Marketing, Manufacturing, Legal, Finance
  rating         FLOAT     -- vendor performance score (1.0 to 5.0)
  onboarded_date DATE      -- when the vendor was approved
  contact_email  VARCHAR
  is_active      VARCHAR   -- 'yes' or 'no'

TABLE: purchase_orders
  po_id       INT PRIMARY KEY
  vendor_id   INT FK → vendors.vendor_id
  amount      FLOAT   -- PO value in USD
  currency    VARCHAR -- always 'USD'
  status      VARCHAR -- one of: open, closed, pending, cancelled
  po_date     DATE    -- date the PO was raised
  category    VARCHAR -- mirrors the vendor category
  description VARCHAR

TABLE: invoices
  invoice_id  INT PRIMARY KEY
  po_id       INT FK → purchase_orders.po_id
  amount      FLOAT
  due_date    DATE
  paid_date   DATE    -- NULL if not yet paid
  status      VARCHAR -- one of: paid, unpaid, overdue, disputed

TABLE: spend_analysis
  id            INT PRIMARY KEY
  category      VARCHAR
  month         VARCHAR   -- format: 'YYYY-MM'
  total_spend   FLOAT
  vendor_count  INT
  avg_po_value  FLOAT
  po_count      INT

Key relationships:
  vendors → purchase_orders (one vendor, many POs)
  purchase_orders → invoices (one PO, many invoices)

Always use table aliases in JOINs. Always limit results to 50 rows unless the user asks for all.
Return results as clean JSON-serialisable data.
"""


if __name__ == "__main__":
    result = check_connection()
    print(result)
