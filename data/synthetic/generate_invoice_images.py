"""
data/synthetic/generate_invoice_images.py
────────────────────────────────────────────
Generates synthetic invoice/PO photos for testing the Vision Agent
(src/agents/vision_agent.py) and its structured-extraction accuracy.

Mirrors generate_erp_data.py's synthetic-data philosophy: realistic-looking
data instead of real vendor invoices, which don't exist for this project
and couldn't be shared even if they did (unlike the contract/policy PDFs,
which are real public-domain templates — see data/contracts/DOWNLOAD_INSTRUCTIONS.md).

Three categories are generated, matching the real testing needs of the
Vision Agent + hybrid SQL cross-check flow:

  matching/    — invoice fields exactly match a real seeded PO/vendor row
                 in Postgres (pulled live from the DB). Use these to test
                 "does this match our records?" end-to-end.
  mismatched/  — same real seeded PO/vendor/invoice reference, but the
                 printed total is deliberately wrong. Use these to confirm
                 the assistant flags a discrepancy instead of agreeing
                 blindly. (Also pulled live from Postgres — see note below.)
  no_match/    — a fictional vendor/PO that doesn't exist in the ERP at
                 all. Use these to confirm the assistant says "not found"
                 rather than hallucinating a match.

Each image is rendered as a clean digital invoice layout, then given a
slight rotation + JPEG recompression to roughly simulate a phone photo of
a printed page. This is intentionally "good enough" for functional VLM
testing, not a high-fidelity photo simulator.

A manifest.json ground-truth file is written alongside the images —
consumed by src/evaluation/vision_eval.py to score extraction accuracy,
the vision-side equivalent of the RAGAS eval for the RAG agent.

Run:
    # Needs Postgres seeded first (see generate_erp_data.py) for both
    # matching/ and mismatched/ — both need real rows to be meaningful:
    python data/synthetic/generate_invoice_images.py

    # Or skip both DB-dependent categories and only generate no_match/:
    python data/synthetic/generate_invoice_images.py --no-db
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from datetime import timedelta

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from faker import Faker
from PIL import Image, ImageDraw, ImageFont
from loguru import logger

random.seed(7)
fake = Faker()
Faker.seed(7)

OUT_DIR = ROOT / "data" / "synthetic" / "invoices"


# ── Rendering ─────────────────────────────────────────────────────────────────

def _load_font(size: int):
    """
    Falls back to PIL's built-in bitmap font if no TrueType font is found,
    so this script stays runnable without any extra system font dependency.
    """
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size)
            except Exception:
                pass
    return ImageFont.load_default()


def render_invoice_image(
    vendor_name   : str,
    po_number     : str,
    invoice_number: str,
    invoice_date  : str,
    line_items    : list,
    total_amount  : float,
    out_path      : Path,
):
    """Draws a simple, clean invoice layout, then adds light photo-like noise."""
    W, H = 850, 1100
    img  = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)

    font_title = _load_font(28)
    font_head  = _load_font(16)
    font_body  = _load_font(14)

    y = 40
    draw.text((40, y), "INVOICE", font=font_title, fill="black"); y += 50
    draw.text((40, y), f"Vendor: {vendor_name}", font=font_head, fill="black"); y += 28
    draw.text((40, y), f"PO Number: {po_number}", font=font_head, fill="black"); y += 28
    draw.text((40, y), f"Invoice Number: {invoice_number}", font=font_head, fill="black"); y += 28
    draw.text((40, y), f"Invoice Date: {invoice_date}", font=font_head, fill="black"); y += 40

    draw.line((40, y, W - 40, y), fill="black", width=2); y += 20
    draw.text((40, y),  "Description", font=font_head, fill="black")
    draw.text((520, y), "Qty",         font=font_head, fill="black")
    draw.text((600, y), "Unit Price",  font=font_head, fill="black")
    draw.text((730, y), "Total",       font=font_head, fill="black")
    y += 24
    draw.line((40, y, W - 40, y), fill="black", width=1); y += 12

    for item in line_items:
        draw.text((40, y),  item["description"][:45],       font=font_body, fill="black")
        draw.text((520, y), str(item["quantity"]),           font=font_body, fill="black")
        draw.text((600, y), f"${item['unit_price']:,.2f}",   font=font_body, fill="black")
        draw.text((730, y), f"${item['total']:,.2f}",        font=font_body, fill="black")
        y += 26

    y += 20
    draw.line((40, y, W - 40, y), fill="black", width=2); y += 20
    draw.text((600, y), "TOTAL:",              font=font_head, fill="black")
    draw.text((730, y), f"${total_amount:,.2f}", font=font_head, fill="black")

    # Light "photo of a printed page" simulation — real invoice photos are
    # rarely perfectly axis-aligned or lossless, and a VLM should be tested
    # against that, not a pixel-perfect digital render.
    img = img.rotate(random.uniform(-1.2, 1.2), expand=True, fillcolor="white")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "JPEG", quality=78)


def _line_items_for_total(total: float) -> list:
    """Splits a target total across 1-3 plausible line items."""
    n = random.choice([1, 1, 2, 3])
    if n == 1:
        return [{
            "description": fake.bs().capitalize(), "quantity": 1,
            "unit_price": round(total, 2), "total": round(total, 2),
        }]
    splits    = sorted(random.sample(range(1, 100), n - 1))
    fractions = [b - a for a, b in zip([0] + splits, splits + [100])]
    items, running = [], 0.0
    for i, frac in enumerate(fractions):
        amt = round(total * frac / 100, 2) if i < n - 1 else round(total - running, 2)
        running += amt
        qty = random.choice([1, 1, 2, 5])
        items.append({
            "description": fake.bs().capitalize(), "quantity": qty,
            "unit_price": round(amt / qty, 2), "total": amt,
        })
    return items


# ── Category generators ────────────────────────────────────────────────────────

def generate_matching(n: int, manifest: list) -> set:
    """
    Pulls real seeded PO/vendor/invoice rows from Postgres, so these images
    are guaranteed to cross-check successfully against the ERP data.

    Returns the set of invoice_ids used, so generate_mismatched() can avoid
    picking the same invoice (which would otherwise produce a matching/
    image and a mismatched/ image quoting two different totals for the
    same invoice number — confusing rather than a real functional bug, but
    easy to avoid).
    """
    try:
        from sqlalchemy import create_engine, text
        database_url = os.getenv(
            "DATABASE_URL", "postgresql://erp_user:erp_pass@localhost:5432/erp_procurement"
        )
        engine = create_engine(database_url)
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT i.invoice_id, i.amount AS invoice_amount, i.due_date,
                       p.po_id, v.name AS vendor_name
                FROM invoices i
                JOIN purchase_orders p ON i.po_id = p.po_id
                JOIN vendors v ON p.vendor_id = v.vendor_id
                ORDER BY random()
                LIMIT :n
            """), {"n": n}).mappings().all()
    except Exception as e:
        logger.warning(
            f"  [matching] Could not reach Postgres ({e}) — skipping matching/ "
            f"(seed the DB first: python data/synthetic/generate_erp_data.py)"
        )
        return set()

    for row in rows:
        po_number      = f"PO-{row['po_id']:05d}"
        invoice_number = f"INV-{row['invoice_id']:05d}"
        total          = float(row["invoice_amount"])
        invoice_date   = (row["due_date"] - timedelta(days=random.randint(10, 30))).isoformat()
        line_items     = _line_items_for_total(total)

        out_path = OUT_DIR / "matching" / f"invoice_{invoice_number}.jpg"
        render_invoice_image(row["vendor_name"], po_number, invoice_number,
                              invoice_date, line_items, total, out_path)

        manifest.append({
            "file"          : str(out_path.relative_to(ROOT)),
            "category"      : "matching",
            "expected_match": True,
            "ground_truth"  : {
                "vendor_name": row["vendor_name"], "po_number": po_number,
                "invoice_number": invoice_number, "invoice_date": invoice_date,
                "total_amount": total, "line_items": line_items,
            },
        })
    logger.success(f"  ✓ Generated {len(rows)} matching/ invoice images from real ERP rows")
    return {row["invoice_id"] for row in rows}


def generate_mismatched(n: int, manifest: list, exclude_invoice_ids: set = frozenset()):
    """
    Pulls a real seeded PO/vendor/invoice row from Postgres — same as
    generate_matching() — but prints a deliberately wrong total. This is
    what actually exercises the "record exists, but the number is wrong"
    discrepancy-flagging path: the vendor/PO/invoice references must be
    real, or the SQL agent has nothing to find a mismatch against and this
    category degenerates into a second no_match/ case.

    (Previously this fabricated a fake vendor + a PO/invoice number outside
    the real seeded ID range (1-500ish), so the SQL agent could never find
    a row at all — every "mismatched" query silently fell through to "no
    records found", identical to no_match/. Fixed to reuse real rows.)

    exclude_invoice_ids: invoice_ids already used by generate_matching(),
    so the same invoice never appears in both matching/ and mismatched/
    with two different "true" totals.
    """
    try:
        from sqlalchemy import create_engine, text
        database_url = os.getenv(
            "DATABASE_URL", "postgresql://erp_user:erp_pass@localhost:5432/erp_procurement"
        )
        engine = create_engine(database_url)
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT i.invoice_id, i.amount AS invoice_amount, i.due_date,
                       p.po_id, v.name AS vendor_name
                FROM invoices i
                JOIN purchase_orders p ON i.po_id = p.po_id
                JOIN vendors v ON p.vendor_id = v.vendor_id
                WHERE i.invoice_id NOT IN :excluded
                ORDER BY random()
                LIMIT :n
            """), {
                "n": n,
                # asyncpg/psycopg2 need a non-empty tuple for NOT IN; (-1,) never matches a real id
                "excluded": tuple(exclude_invoice_ids) if exclude_invoice_ids else (-1,),
            }).mappings().all()
    except Exception as e:
        logger.warning(
            f"  [mismatched] Could not reach Postgres ({e}) — skipping mismatched/ "
            f"(seed the DB first: python data/synthetic/generate_erp_data.py)"
        )
        return

    for row in rows:
        po_number      = f"PO-{row['po_id']:05d}"
        invoice_number = f"INV-{row['invoice_id']:05d}"
        real_total     = float(row["invoice_amount"])
        printed_total  = round(real_total * random.choice([1.15, 0.85, 1.3]), 2)
        invoice_date   = (row["due_date"] - timedelta(days=random.randint(10, 30))).isoformat()
        line_items     = _line_items_for_total(printed_total)

        out_path = OUT_DIR / "mismatched" / f"invoice_{invoice_number}.jpg"
        render_invoice_image(row["vendor_name"], po_number, invoice_number,
                              invoice_date, line_items, printed_total, out_path)

        manifest.append({
            "file"          : str(out_path.relative_to(ROOT)),
            "category"      : "mismatched",
            "expected_match": False,
            "note"          : (
                f"Real ERP total for this invoice is ${real_total:,.2f}; printed "
                f"total (${printed_total:,.2f}) is intentionally wrong."
            ),
            "ground_truth"  : {
                "vendor_name": row["vendor_name"], "po_number": po_number,
                "invoice_number": invoice_number, "invoice_date": invoice_date,
                "total_amount": printed_total, "line_items": line_items,
            },
        })
    logger.success(f"  ✓ Generated {len(rows)} mismatched/ invoice images from real ERP rows (wrong total printed)")


def generate_no_match(n: int, manifest: list):
    """
    A fictional vendor/PO that doesn't exist in the ERP at all — tests that
    the assistant says "not found" rather than hallucinating a match.
    """
    for _ in range(n):
        vendor_name    = fake.company() + " (Not In ERP)"
        po_number      = f"PO-{random.randint(90000, 99999)}"
        invoice_number = f"INV-{random.randint(90000, 99999)}"
        total          = round(random.uniform(500, 20000), 2)
        invoice_date   = fake.date_between(start_date="-30d", end_date="today").isoformat()
        line_items     = _line_items_for_total(total)

        out_path = OUT_DIR / "no_match" / f"invoice_{invoice_number}.jpg"
        render_invoice_image(vendor_name, po_number, invoice_number,
                              invoice_date, line_items, total, out_path)

        manifest.append({
            "file"          : str(out_path.relative_to(ROOT)),
            "category"      : "no_match",
            "expected_match": None,
            "ground_truth"  : {
                "vendor_name": vendor_name, "po_number": po_number,
                "invoice_number": invoice_number, "invoice_date": invoice_date,
                "total_amount": total, "line_items": line_items,
            },
        })
    logger.success(f"  ✓ Generated {n} no_match/ invoice images")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-db", action="store_true",
                         help="Skip the matching/ category (no Postgres connection needed)")
    parser.add_argument("--n-matching",   type=int, default=5)
    parser.add_argument("--n-mismatched", type=int, default=3)
    parser.add_argument("--n-no-match",   type=int, default=2)
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("GENERATING SYNTHETIC INVOICE/PO IMAGES (for Vision Agent testing)")
    print("=" * 60)

    manifest = []
    used_invoice_ids = set()
    if not args.no_db:
        used_invoice_ids = generate_matching(args.n_matching, manifest)
        generate_mismatched(args.n_mismatched, manifest, exclude_invoice_ids=used_invoice_ids)
    else:
        logger.warning(
            "  --no-db set — skipping matching/ AND mismatched/ (both need real "
            "Postgres rows to be meaningful; only no_match/ works without a DB)"
        )
    generate_no_match(args.n_no_match, manifest)

    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"\n✓ {len(manifest)} images generated under {OUT_DIR.relative_to(ROOT)}/")
    print(f"✓ Ground truth manifest written to {manifest_path.relative_to(ROOT)}")
    print("\nTry one in Streamlit, or run the accuracy eval:")
    print("  python src/evaluation/vision_eval.py")


if __name__ == "__main__":
    main()
