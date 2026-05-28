"""
data/contracts/create_sample_contracts.py
──────────────────────────────────────────
Generates 3 minimal synthetic contract PDFs for pipeline testing.
Run this if you want to test the ingestion pipeline immediately
without downloading the real UNOPS / World Bank documents.

Output:
    data/contracts/vendor_contract_alpha_tech.pdf
    data/contracts/vendor_contract_beta_logistics.pdf
    data/contracts/vendor_contract_gamma_consulting.pdf
    data/policies/internal_procurement_policy.pdf

Usage:
    python data/contracts/create_sample_contracts.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.enums import TA_JUSTIFY
except ImportError:
    print("reportlab not installed. Run: pip install reportlab")
    sys.exit(1)

from loguru import logger


STYLES = getSampleStyleSheet()
BODY   = ParagraphStyle("body", parent=STYLES["Normal"], fontSize=10,
                         spaceAfter=8, leading=14, alignment=TA_JUSTIFY)
H1     = ParagraphStyle("h1",   parent=STYLES["Heading1"], fontSize=14, spaceAfter=12)
H2     = ParagraphStyle("h2",   parent=STYLES["Heading2"], fontSize=11, spaceAfter=8)


def make_pdf(path: Path, title: str, sections: list[tuple[str, str]]):
    doc = SimpleDocTemplate(str(path), pagesize=A4,
                            leftMargin=3*cm, rightMargin=3*cm,
                            topMargin=3*cm,  bottomMargin=3*cm)
    story = []
    story.append(Paragraph(title, H1))
    story.append(Spacer(1, 0.5*cm))

    for heading, body_text in sections:
        story.append(Paragraph(heading, H2))
        for para in body_text.strip().split("\n\n"):
            story.append(Paragraph(para.strip(), BODY))
        story.append(Spacer(1, 0.3*cm))

    doc.build(story)
    logger.success(f"Created: {path}")


# ── Contract 1 — Alpha Tech Solutions ────────────────────────────────────────

CONTRACT_ALPHA = [
    ("1. Parties", """
This Services Agreement ("Agreement") is entered into between Alpha Tech Solutions
("Vendor") and the Buyer Organisation ("Buyer") as of the Effective Date set forth below.
The Vendor agrees to provide IT infrastructure management and cloud migration services
as described in Schedule A attached hereto.
"""),
    ("2. Payment Terms", """
The Buyer shall pay the Vendor within thirty (30) days of receipt of a valid invoice.
Invoices submitted after the 25th of any calendar month shall be processed in the
following payment cycle. All amounts are stated in United States Dollars (USD).

Late payment by the Buyer shall accrue interest at a rate of 1.5% per month on the
outstanding balance. The Vendor shall provide written notice of overdue amounts within
fifteen (15) days of the payment due date.
"""),
    ("3. Delivery and Performance Standards", """
The Vendor shall deliver all contracted services within the timelines specified in
Schedule B. Time is of the essence for all deliverables.

In the event of a Late Delivery, defined as any deliverable not completed within
five (5) business days of the agreed deadline, the Buyer may apply a Late Delivery
Penalty. The Late Delivery Penalty shall be calculated as 0.5% of the total contract
value for each week of delay, up to a maximum of 10% of the total contract value.

The Vendor shall notify the Buyer in writing within 24 hours of identifying any risk
of late delivery, including the cause and expected revised completion date.
"""),
    ("4. Intellectual Property", """
All work product, deliverables, and documentation produced by the Vendor under this
Agreement shall become the exclusive property of the Buyer upon full payment. The
Vendor retains no license or rights to use such materials for any other purpose
without prior written consent of the Buyer.
"""),
    ("5. Confidentiality", """
Both parties agree to maintain strict confidentiality of all proprietary information
exchanged under this Agreement. Confidential information shall not be disclosed to
any third party without prior written approval. This obligation survives termination
of this Agreement for a period of five (5) years.
"""),
    ("6. Termination for Convenience", """
Either party may terminate this Agreement for convenience upon sixty (60) days written
notice to the other party. In the event of termination for convenience by the Buyer,
the Vendor shall be compensated for all work performed up to the date of termination
plus reasonable demobilisation costs not exceeding 10% of the remaining contract value.

Termination for Cause may be exercised immediately upon written notice if the other
party commits a material breach and fails to remedy such breach within thirty (30)
days of receiving notice.
"""),
    ("7. Force Majeure", """
Neither party shall be liable for delays or failures in performance resulting from
circumstances beyond their reasonable control, including acts of God, war, terrorism,
pandemic, government action, or natural disaster ("Force Majeure Event").

The affected party must notify the other party within five (5) business days of the
Force Majeure Event and provide a remediation plan within fifteen (15) days.
If a Force Majeure Event continues for more than ninety (90) days, either party may
terminate this Agreement without penalty.
"""),
    ("8. Dispute Resolution", """
Any disputes arising from this Agreement shall first be subject to good-faith
negotiation for thirty (30) days. If unresolved, disputes shall be submitted to
binding arbitration under the rules of the International Chamber of Commerce.
"""),
    ("9. Governing Law", """
This Agreement shall be governed by and construed in accordance with the laws of
the State of New York, without regard to its conflict of laws principles.
"""),
]

# ── Contract 2 — Beta Logistics Group ────────────────────────────────────────

CONTRACT_BETA = [
    ("1. Scope of Services", """
Beta Logistics Group ("Vendor") agrees to provide third-party logistics, warehousing,
and freight forwarding services to the Buyer as detailed in Annex 1. Services include
last-mile delivery, customs clearance support, and real-time shipment tracking.
"""),
    ("2. Pricing and Invoicing", """
Services shall be invoiced monthly based on actual volumes delivered, per the rate
schedule in Annex 2. Net payment terms are forty-five (45) days from invoice date.
The Vendor may not increase rates during the contract term without sixty (60) days
prior written notice and Buyer approval.
"""),
    ("3. Service Level Agreement and Penalties", """
The Vendor commits to the following Service Level Agreement (SLA):
  - On-time delivery rate: minimum 95% of shipments delivered within agreed window
  - Damage rate: maximum 0.2% of total shipments per quarter
  - Claims response time: within 48 hours of reported incident

SLA Penalty Structure:
If on-time delivery falls below 95% in any calendar month, the Buyer may deduct
a Service Credit of 2% of that month's invoiced amount for each percentage point
below the threshold. Total monthly service credits shall not exceed 15% of the
monthly invoice value.

Repeated failure to meet SLA targets for three (3) consecutive months shall
constitute material breach, entitling the Buyer to terminate for cause.
"""),
    ("4. Insurance Requirements", """
The Vendor shall maintain, at its own cost, the following minimum insurance coverage:
  - Commercial General Liability: USD 5,000,000 per occurrence
  - Cargo Insurance: USD 2,000,000 per shipment
  - Workers Compensation: as required by applicable law

Certificates of insurance shall be provided to the Buyer annually and upon request.
"""),
    ("5. Data and Systems", """
The Vendor shall provide real-time API access to shipment tracking data. All data
exchanged under this Agreement remains the property of the Buyer. The Vendor shall
implement industry-standard data security measures and notify the Buyer within
24 hours of any data breach affecting Buyer data.
"""),
    ("6. Environmental Compliance", """
The Vendor commits to measuring and reporting carbon emissions associated with
Buyer shipments on a quarterly basis. The Vendor shall develop a Carbon Reduction
Plan within 90 days of contract execution, targeting a 15% reduction over the
contract term.
"""),
    ("7. Subcontracting", """
The Vendor may not subcontract more than 30% of the contracted services without
prior written approval from the Buyer. Approved subcontractors are bound by the
same obligations, standards, and confidentiality requirements as the Vendor.
The Vendor remains fully liable for any acts or omissions of its subcontractors.
"""),
]

# ── Contract 3 — Gamma Consulting Partners ────────────────────────────────────

CONTRACT_GAMMA = [
    ("1. Engagement Scope", """
Gamma Consulting Partners ("Consultant") is engaged to provide management consulting
services in the areas of procurement transformation, spend analytics, and supplier
relationship management. Services are detailed in the Statement of Work ("SOW")
attached as Exhibit A.
"""),
    ("2. Fees and Expenses", """
Consulting fees are billed at the daily rates specified in Exhibit B, based on
actual days worked. Expenses are reimbursable at cost with receipts, subject to
the Buyer's Travel and Expense Policy. Invoices are due net thirty (30) days.
"""),
    ("3. Key Personnel", """
The Consultant shall assign the personnel named in Exhibit C as Key Personnel
for this engagement. Key Personnel may not be replaced without forty-five (45)
days prior written notice and Buyer approval of the replacement. Unauthorised
replacement of Key Personnel constitutes a material breach.
"""),
    ("4. Deliverables and Acceptance", """
Each deliverable is subject to a Buyer review period of fifteen (15) business days.
If the Buyer does not provide written comments within the review period, the
deliverable is deemed accepted. The Consultant shall address all material comments
within ten (10) business days of receipt.
"""),
    ("5. Non-Solicitation", """
During the contract term and for twelve (12) months thereafter, neither party shall
solicit or hire any employee of the other party who was involved in the provision
or receipt of services under this Agreement.
"""),
    ("6. Limitation of Liability", """
In no event shall either party be liable for indirect, incidental, special, or
consequential damages. The Consultant's total liability under this Agreement shall
not exceed the total fees paid in the six (6) months preceding the claim.
"""),
    ("7. Warranties", """
The Consultant warrants that: (a) services will be performed in a professional and
workmanlike manner consistent with industry standards; (b) all personnel have the
qualifications represented; (c) services do not infringe any third-party rights.
"""),
]

# ── Policy Document ───────────────────────────────────────────────────────────

POLICY_SECTIONS = [
    ("1. Purpose and Scope", """
This Procurement Policy governs all purchasing activities of the Organisation.
It applies to all employees, contractors, and agents involved in procurement
decisions. Compliance is mandatory. Exceptions require written approval from
the Chief Procurement Officer.
"""),
    ("2. Procurement Thresholds and Approval Levels", """
All procurement activities must follow the approval authority matrix below:

  - Up to USD 10,000        : Department Manager approval
  - USD 10,001 – 50,000     : Senior Manager + Finance approval
  - USD 50,001 – 250,000    : Director approval + Procurement review
  - USD 250,001 – 1,000,000 : VP approval + Legal review + Procurement
  - Above USD 1,000,000     : C-suite approval + Board notification

Purchase requests above USD 100,000 require a minimum of three (3) competitive
bids documented in the procurement file. Single-source justification is only
permissible above USD 100,000 with documented approval from the Chief Procurement
Officer citing one of the approved exceptions (sole supplier, emergency, proprietary).
"""),
    ("3. Vendor Selection and Evaluation", """
Vendors must be registered in the Approved Vendor Registry before any PO is issued.
New vendor onboarding requires completion of a due diligence questionnaire, financial
stability review, and sanctions screening.

Vendor evaluation criteria shall include: technical capability (40%), price (30%),
delivery reliability (20%), and sustainability practices (10%). Deviations from
this weighting require documented justification.
"""),
    ("4. Conflict of Interest", """
All procurement staff must declare any personal, financial, or professional
relationship with vendors or bidders prior to participating in any evaluation,
selection, or approval process involving that party.

Undisclosed conflicts of interest constitute grounds for disciplinary action
including termination. Declared conflicts require recusal from the relevant
procurement decision.
"""),
    ("5. Single-Source and Sole-Source Procurement", """
Single-source procurement — awarding a contract without competitive bidding — is
prohibited above USD 100,000 except in the following circumstances:

  a) Only one supplier is technically capable of providing the goods or services
  b) Emergency situation where competitive bidding would cause unacceptable delay
  c) Proprietary technology where only one vendor holds the rights
  d) Follow-on procurement where changing supplier would cause significant cost

All single-source awards above USD 100,000 must be approved by the CPO and
documented with a written justification retained in the procurement file for audit.
"""),
    ("6. Contract Management", """
All contracts above USD 25,000 must be executed using the Organisation's standard
contract templates. Deviations require Legal review and approval.

Contract performance must be formally reviewed at least annually. Vendors with
performance ratings below 3.0 out of 5.0 for two consecutive reviews are placed
on the Underperformance Watch List and may not receive new POs until remediation.
"""),
    ("7. Maverick Spend", """
Maverick spend — purchasing without following procurement policy — is a compliance
breach. All identified maverick spend must be reported to the Procurement function
within 30 days of discovery.

Departments with maverick spend exceeding 5% of annual procurement budget are
subject to mandatory procurement process audit and corrective action planning.
"""),
]


def main():
    contracts_dir = ROOT / "data" / "contracts"
    policies_dir  = ROOT / "data" / "policies"
    contracts_dir.mkdir(parents=True, exist_ok=True)
    policies_dir.mkdir(parents=True, exist_ok=True)

    make_pdf(
        contracts_dir / "vendor_contract_alpha_tech.pdf",
        "SERVICES AGREEMENT — ALPHA TECH SOLUTIONS",
        CONTRACT_ALPHA,
    )
    make_pdf(
        contracts_dir / "vendor_contract_beta_logistics.pdf",
        "LOGISTICS SERVICES AGREEMENT — BETA LOGISTICS GROUP",
        CONTRACT_BETA,
    )
    make_pdf(
        contracts_dir / "vendor_contract_gamma_consulting.pdf",
        "CONSULTING SERVICES AGREEMENT — GAMMA CONSULTING PARTNERS",
        CONTRACT_GAMMA,
    )
    make_pdf(
        policies_dir / "internal_procurement_policy.pdf",
        "ORGANISATION PROCUREMENT POLICY — VERSION 4.2",
        POLICY_SECTIONS,
    )

    logger.success("\nAll sample PDFs created. Run ingestion pipeline:")
    logger.success("  python src/ingestion/ingest_pipeline.py --all")


if __name__ == "__main__":
    main()
