"""
data/synthetic/generate_erp_data.py
─────────────────────────────────────
Generates realistic synthetic ERP data and inserts it into PostgreSQL.

Run:
    python data/synthetic/generate_erp_data.py

Tables populated
----------------
  vendors         : ~200 rows
  purchase_orders : ~500 rows
  invoices        : ~400 rows
  spend_analysis  : ~100 rows (derived from POs)
"""

import os
import sys
import random
from datetime import date, timedelta
from pathlib import Path

# ── Path fix so we can import from src/ ───────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
from faker import Faker
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from loguru import logger

from src.data.schema import Base, Vendor, PurchaseOrder, Invoice, SpendAnalysis

load_dotenv()

fake = Faker()
random.seed(42)
Faker.seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
DATABASE_URL  = os.getenv("DATABASE_URL",
    "postgresql://erp_user:erp_pass@localhost:5432/erp_procurement")

N_VENDORS = 200
N_POS     = 500
N_INVOICES_RATIO = 0.80   # 80% of POs get at least one invoice

CATEGORIES = [
    "IT Services", "Logistics", "Consulting", "Facilities",
    "Marketing", "Manufacturing", "Legal", "Finance"
]

COUNTRIES = [
    "United States", "Germany", "India", "United Kingdom",
    "France", "Canada", "Singapore", "Netherlands",
    "Australia", "Japan", "Brazil", "Sweden"
]

PO_STATUS_WEIGHTS  = ["open", "open", "closed", "closed", "pending", "cancelled"]
INV_STATUS_MAP     = {
    "open"      : ["unpaid", "overdue", "disputed"],
    "closed"    : ["paid", "paid", "paid"],
    "pending"   : ["unpaid"],
    "cancelled" : [],        # no invoices for cancelled POs
}

CURRENCY = "USD"


# ── Helpers ───────────────────────────────────────────────────────────────────

def rand_date(start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))


def make_vendor_name() -> str:
    """Generate a company name that sounds like a real enterprise vendor."""
    suffixes = ["Group", "Solutions", "Partners", "Global", "Services",
                "Consulting", "Systems", "Technologies", "Ventures", "Corp"]
    return f"{fake.last_name()} {random.choice(suffixes)}"


# ── Generator functions ───────────────────────────────────────────────────────

def generate_vendors(n: int = N_VENDORS) -> list[Vendor]:
    logger.info(f"Generating {n} vendors...")
    vendors = []
    names_used = set()

    for _ in range(n):
        name = make_vendor_name()
        while name in names_used:
            name = make_vendor_name()
        names_used.add(name)

        vendors.append(Vendor(
            name           = name,
            country        = random.choice(COUNTRIES),
            category       = random.choice(CATEGORIES),
            rating         = round(random.uniform(2.5, 5.0), 1),
            onboarded_date = rand_date(date(2018, 1, 1), date(2023, 12, 31)),
            contact_email  = fake.company_email(),
            is_active      = random.choices(["yes", "no"], weights=[90, 10])[0],
        ))
    return vendors


def generate_purchase_orders(vendors: list[Vendor], n: int = N_POS) -> list[PurchaseOrder]:
    logger.info(f"Generating {n} purchase orders...")
    pos = []
    active_vendors = [v for v in vendors if v.is_active == "yes"]

    for _ in range(n):
        vendor   = random.choice(active_vendors)
        status   = random.choice(PO_STATUS_WEIGHTS)
        category = vendor.category          # PO category mirrors vendor category

        # Amount distribution: realistic enterprise spend range
        amount_tier = random.choices(
            ["small", "medium", "large", "strategic"],
            weights=[40, 35, 20, 5]
        )[0]
        amount = {
            "small"    : round(random.uniform(1_000,   25_000),  2),
            "medium"   : round(random.uniform(25_000,  150_000), 2),
            "large"    : round(random.uniform(150_000, 500_000), 2),
            "strategic": round(random.uniform(500_000, 2_000_000), 2),
        }[amount_tier]

        po_date = rand_date(date(2023, 1, 1), date(2024, 9, 30))

        pos.append(PurchaseOrder(
            vendor_id   = vendor.vendor_id,
            amount      = amount,
            currency    = CURRENCY,
            status      = status,
            po_date     = po_date,
            category    = category,
            description = fake.bs().capitalize(),
        ))
    return pos


def generate_invoices(pos: list[PurchaseOrder]) -> list[Invoice]:
    logger.info("Generating invoices...")
    invoices = []

    for po in pos:
        possible_statuses = INV_STATUS_MAP.get(po.status, [])
        if not possible_statuses:
            continue
        if random.random() > N_INVOICES_RATIO:
            continue

        # Some POs get split invoices (2–3 partial payments)
        n_invoices = random.choices([1, 2, 3], weights=[70, 20, 10])[0]
        amounts    = _split_amount(po.amount, n_invoices)

        for inv_amount in amounts:
            status   = random.choice(possible_statuses)
            due_date = po.po_date + timedelta(days=random.randint(30, 90))

            paid_date = None
            if status == "paid":
                paid_date = due_date - timedelta(days=random.randint(0, 15))
            elif status == "overdue":
                # overdue = due date already passed, not paid
                due_date = date.today() - timedelta(days=random.randint(5, 120))

            invoices.append(Invoice(
                po_id     = po.po_id,
                amount    = inv_amount,
                due_date  = due_date,
                paid_date = paid_date,
                status    = status,
            ))
    return invoices


def _split_amount(total: float, n: int) -> list[float]:
    """Split a total amount into n realistic partial payments."""
    if n == 1:
        return [total]
    cuts = sorted(random.sample(range(1, 100), n - 1))
    cuts = [0] + cuts + [100]
    return [round(total * (cuts[i+1] - cuts[i]) / 100, 2) for i in range(n)]


def generate_spend_analysis(pos: list[PurchaseOrder]) -> list[SpendAnalysis]:
    """Aggregate POs into monthly spend-by-category rows."""
    logger.info("Generating spend_analysis...")

    records = []
    for po in pos:
        if po.status == "cancelled":
            continue
        records.append({
            "category": po.category,
            "month"   : po.po_date.strftime("%Y-%m"),
            "amount"  : po.amount,
        })

    df = pd.DataFrame(records)
    grouped = df.groupby(["category", "month"]).agg(
        total_spend  = ("amount", "sum"),
        vendor_count = ("amount", "count"),
        avg_po_value = ("amount", "mean"),
        po_count     = ("amount", "count"),
    ).reset_index()

    rows = []
    for _, row in grouped.iterrows():
        rows.append(SpendAnalysis(
            category     = row["category"],
            month        = row["month"],
            total_spend  = round(row["total_spend"],  2),
            vendor_count = int(row["vendor_count"]),
            avg_po_value = round(row["avg_po_value"], 2),
            po_count     = int(row["po_count"]),
        ))
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    logger.info("Connecting to database...")
    engine  = create_engine(DATABASE_URL, echo=False)

    logger.info("Creating tables (if not exist)...")
    Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Safety: skip if data already loaded
        if session.query(Vendor).count() > 0:
            logger.warning("Data already exists. Drop tables first if you want to regenerate.")
            logger.info("Tip: docker-compose down -v && docker-compose up -d")
            return

        # 1. Vendors
        vendors = generate_vendors()
        session.bulk_save_objects(vendors)
        session.flush()
        # Reload to get DB-assigned IDs
        vendors = session.query(Vendor).all()
        logger.success(f"  ✓ {len(vendors)} vendors inserted")

        # 2. Purchase Orders
        pos = generate_purchase_orders(vendors)
        session.bulk_save_objects(pos)
        session.flush()
        pos = session.query(PurchaseOrder).all()
        logger.success(f"  ✓ {len(pos)} purchase orders inserted")

        # 3. Invoices
        invoices = generate_invoices(pos)
        session.bulk_save_objects(invoices)
        session.flush()
        logger.success(f"  ✓ {len(invoices)} invoices inserted")

        # 4. Spend Analysis
        spend_rows = generate_spend_analysis(pos)
        session.bulk_save_objects(spend_rows)
        session.flush()
        logger.success(f"  ✓ {len(spend_rows)} spend_analysis rows inserted")

        session.commit()
        logger.success("All data committed successfully.")

        # ── Summary ───────────────────────────────────────────────────────────
        print("\n" + "="*50)
        print("DATABASE SUMMARY")
        print("="*50)
        print(f"  vendors         : {session.query(Vendor).count()}")
        print(f"  purchase_orders : {session.query(PurchaseOrder).count()}")
        print(f"  invoices        : {session.query(Invoice).count()}")
        print(f"  spend_analysis  : {session.query(SpendAnalysis).count()}")
        print("="*50)

        # ── Sample data preview ───────────────────────────────────────────────
        print("\nSAMPLE VENDORS:")
        for v in session.query(Vendor).limit(3).all():
            print(f"  {v}")

        print("\nSAMPLE POs:")
        for po in session.query(PurchaseOrder).limit(3).all():
            print(f"  {po}")

        print("\nSAMPLE INVOICES:")
        for inv in session.query(Invoice).limit(3).all():
            print(f"  {inv}")

    except Exception as e:
        session.rollback()
        logger.error(f"Error: {e}")
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()
