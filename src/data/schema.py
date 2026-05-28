"""
src/data/schema.py
──────────────────
SQLAlchemy ORM models for the ERP procurement database.

Tables
------
  vendors          — supplier master data
  purchase_orders  — PO header records
  invoices         — invoice records tied to POs
  spend_analysis   — pre-aggregated monthly spend (like an ERP reporting table)
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Date, DateTime,
    ForeignKey, Enum, CheckConstraint, Index, text
)
from sqlalchemy.orm import declarative_base, relationship
import enum

Base = declarative_base()


# ── Enums ─────────────────────────────────────────────────────────────────────

class POStatus(str, enum.Enum):
    OPEN       = "open"
    CLOSED     = "closed"
    CANCELLED  = "cancelled"
    PENDING    = "pending"


class InvoiceStatus(str, enum.Enum):
    PAID       = "paid"
    UNPAID     = "unpaid"
    OVERDUE    = "overdue"
    DISPUTED   = "disputed"


class VendorCategory(str, enum.Enum):
    IT_SERVICES    = "IT Services"
    LOGISTICS      = "Logistics"
    CONSULTING     = "Consulting"
    FACILITIES     = "Facilities"
    MARKETING      = "Marketing"
    MANUFACTURING  = "Manufacturing"
    LEGAL          = "Legal"
    FINANCE        = "Finance"


# ── Models ────────────────────────────────────────────────────────────────────

class Vendor(Base):
    """Supplier master record — ~200 rows."""
    __tablename__ = "vendors"

    vendor_id      = Column(Integer, primary_key=True, autoincrement=True)
    name           = Column(String(120), nullable=False, unique=True)
    country        = Column(String(60),  nullable=False)
    category       = Column(String(60),  nullable=False)   # VendorCategory values
    rating         = Column(Float,       nullable=False)    # 1.0 – 5.0
    onboarded_date = Column(Date,        nullable=False)
    contact_email  = Column(String(120), nullable=True)
    is_active      = Column(String(3),   default="yes")    # yes / no

    # Relationships
    purchase_orders = relationship("PurchaseOrder", back_populates="vendor",
                                   cascade="all, delete-orphan")

    __table_args__ = (
        CheckConstraint("rating >= 1.0 AND rating <= 5.0", name="ck_vendor_rating"),
        Index("ix_vendors_category", "category"),
        Index("ix_vendors_country",  "country"),
    )

    def __repr__(self):
        return f"<Vendor id={self.vendor_id} name='{self.name}' category='{self.category}'>"


class PurchaseOrder(Base):
    """PO header — ~500 rows."""
    __tablename__ = "purchase_orders"

    po_id      = Column(Integer,     primary_key=True, autoincrement=True)
    vendor_id  = Column(Integer,     ForeignKey("vendors.vendor_id"), nullable=False)
    amount     = Column(Float,       nullable=False)
    currency   = Column(String(3),   default="USD")
    status     = Column(String(20),  nullable=False)    # POStatus values
    po_date    = Column(Date,        nullable=False)
    category   = Column(String(60),  nullable=False)   # mirrors VendorCategory
    description= Column(String(255), nullable=True)

    # Relationships
    vendor   = relationship("Vendor",   back_populates="purchase_orders")
    invoices = relationship("Invoice",  back_populates="purchase_order",
                            cascade="all, delete-orphan")

    __table_args__ = (
        CheckConstraint("amount > 0", name="ck_po_amount_positive"),
        Index("ix_po_vendor_id", "vendor_id"),
        Index("ix_po_status",    "status"),
        Index("ix_po_date",      "po_date"),
    )

    def __repr__(self):
        return f"<PO id={self.po_id} vendor_id={self.vendor_id} amount={self.amount} status='{self.status}'>"


class Invoice(Base):
    """Invoice records linked to POs — ~400 rows."""
    __tablename__ = "invoices"

    invoice_id = Column(Integer,  primary_key=True, autoincrement=True)
    po_id      = Column(Integer,  ForeignKey("purchase_orders.po_id"), nullable=False)
    amount     = Column(Float,    nullable=False)
    due_date   = Column(Date,     nullable=False)
    paid_date  = Column(Date,     nullable=True)   # NULL = not yet paid
    status     = Column(String(20), nullable=False) # InvoiceStatus values

    # Relationships
    purchase_order = relationship("PurchaseOrder", back_populates="invoices")

    __table_args__ = (
        CheckConstraint("amount > 0", name="ck_invoice_amount_positive"),
        Index("ix_invoice_po_id",  "po_id"),
        Index("ix_invoice_status", "status"),
        Index("ix_invoice_due",    "due_date"),
    )

    def __repr__(self):
        return f"<Invoice id={self.invoice_id} po_id={self.po_id} amount={self.amount} status='{self.status}'>"


class SpendAnalysis(Base):
    """
    Pre-aggregated monthly spend per category.
    Mirrors what a typical ERP reporting/BI layer would materialise — ~100 rows.
    """
    __tablename__ = "spend_analysis"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    category      = Column(String(60), nullable=False)
    month         = Column(String(7),  nullable=False)   # e.g. "2024-03"
    total_spend   = Column(Float,      nullable=False)
    vendor_count  = Column(Integer,    nullable=False)
    avg_po_value  = Column(Float,      nullable=False)
    po_count      = Column(Integer,    nullable=False)

    __table_args__ = (
        Index("ix_spend_category", "category"),
        Index("ix_spend_month",    "month"),
    )

    def __repr__(self):
        return f"<SpendAnalysis category='{self.category}' month='{self.month}' total={self.total_spend}>"
