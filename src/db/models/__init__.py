"""Every table, imported here so ``Base.metadata`` is complete for Alembic."""

from db.models.transaction import ExtractionStatus, Transaction

__all__ = ["ExtractionStatus", "Transaction"]
