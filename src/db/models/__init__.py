"""Every table, imported here so ``Base.metadata`` is complete for Alembic."""

from db.models.attachment import Attachment
from db.models.message import Message, MessageDirection, MessageStatus, MessageType
from db.models.transaction import ExtractionStatus, PartyRole, Transaction, TransactionParty
from db.models.user import User

__all__ = [
    "Attachment",
    "ExtractionStatus",
    "Message",
    "MessageDirection",
    "MessageStatus",
    "MessageType",
    "PartyRole",
    "Transaction",
    "TransactionParty",
    "User",
]
