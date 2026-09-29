"""initial schema: users, messages, attachments, transactions, transaction_parties

Revision ID: 0001
Revises:
Create Date: 2026-09-29 07:18:20.166844+00:00

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

JSON = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
MONEY = sa.Numeric(precision=18, scale=2)


def str_enum(name: str, *values: str) -> sa.Enum:
    return sa.Enum(*values, name=name, native_enum=False, create_constraint=False, length=16)


def timestamps() -> list[sa.Column]:
    return [
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    ]


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("channel", sa.String(length=32), nullable=False),
        sa.Column("external_id", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=True),
        sa.Column("username", sa.String(length=255), nullable=True),
        *timestamps(),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_users")),
        sa.UniqueConstraint("channel", "external_id", name=op.f("uq_users_channel_external_id")),
    )

    op.create_table(
        "messages",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("channel", sa.String(length=32), nullable=False),
        sa.Column("conversation_id", sa.String(length=64), nullable=False),
        sa.Column("direction", str_enum("messagedirection", "inbound", "outbound"), nullable=False),
        sa.Column("type", str_enum("messagetype", "text", "image"), nullable=False),
        sa.Column(
            "status",
            str_enum("messagestatus", "received", "sent", "delivered", "read", "failed"),
            nullable=False,
        ),
        sa.Column("provider_message_id", sa.String(length=128), nullable=True),
        sa.Column("reply_to_id", sa.Integer(), nullable=True),
        sa.Column("text", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("sent_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("read_at", sa.DateTime(timezone=True), nullable=True),
        *timestamps(),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name=op.f("fk_messages_user_id_users"), ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["reply_to_id"], ["messages.id"], name=op.f("fk_messages_reply_to_id_messages"), ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_messages")),
        sa.UniqueConstraint(
            "channel", "provider_message_id", name=op.f("uq_messages_channel_provider_message_id")
        ),
    )
    op.create_index(op.f("ix_messages_user_id"), "messages", ["user_id"])
    op.create_index(op.f("ix_messages_reply_to_id"), "messages", ["reply_to_id"])
    op.create_index(
        "ix_messages_conversation_created", "messages", ["channel", "conversation_id", "created_at"]
    )

    op.create_table(
        "attachments",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("message_id", sa.Integer(), nullable=False),
        sa.Column("type", str_enum("attachmenttype", "image"), nullable=False),
        sa.Column("provider_ref", sa.String(length=128), nullable=True),
        sa.Column("mime_type", sa.String(length=64), nullable=True),
        sa.Column("filename", sa.String(length=255), nullable=True),
        sa.Column("size_bytes", sa.Integer(), nullable=True),
        sa.Column("sha256", sa.String(length=64), nullable=True),
        sa.Column("data", sa.LargeBinary(), nullable=True),
        sa.Column("storage_path", sa.String(length=1024), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(
            ["message_id"], ["messages.id"], name=op.f("fk_attachments_message_id_messages"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_attachments")),
    )
    op.create_index(op.f("ix_attachments_message_id"), "attachments", ["message_id"])
    op.create_index(op.f("ix_attachments_sha256"), "attachments", ["sha256"])

    op.create_table(
        "transactions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("attachment_id", sa.Integer(), nullable=True),
        sa.Column(
            "extraction_status",
            str_enum("extractionstatus", "pending", "extracted", "no_data", "failed"),
            nullable=False,
        ),
        sa.Column("ocr_engine", sa.String(length=32), nullable=True),
        sa.Column("provider", sa.String(length=64), nullable=True),
        sa.Column("transfer_status", sa.String(length=32), nullable=True),
        sa.Column("transfer_type", sa.String(length=64), nullable=True),
        sa.Column("amount", MONEY, nullable=True),
        sa.Column("fees", MONEY, nullable=True),
        sa.Column("total", MONEY, nullable=True),
        sa.Column("currency", sa.String(length=3), nullable=True),
        sa.Column("reference", sa.String(length=128), nullable=True),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("occurred_at_raw", sa.String(length=64), nullable=True),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("other_data", JSON, nullable=True),
        sa.Column("raw_result", JSON, nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        *timestamps(),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name=op.f("fk_transactions_user_id_users"), ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["attachment_id"],
            ["attachments.id"],
            name=op.f("fk_transactions_attachment_id_attachments"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_transactions")),
        sa.UniqueConstraint("attachment_id", name=op.f("uq_transactions_attachment_id")),
    )
    op.create_index(op.f("ix_transactions_user_id"), "transactions", ["user_id"])
    op.create_index(op.f("ix_transactions_extraction_status"), "transactions", ["extraction_status"])
    op.create_index(op.f("ix_transactions_occurred_at"), "transactions", ["occurred_at"])
    op.create_index("ix_transactions_provider_reference", "transactions", ["provider", "reference"])

    op.create_table(
        "transaction_parties",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("transaction_id", sa.Integer(), nullable=False),
        sa.Column("role", str_enum("partyrole", "sender", "receiver"), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=True),
        sa.Column("name_alt", sa.String(length=255), nullable=True),
        sa.Column("phone", sa.String(length=32), nullable=True),
        sa.Column("email", sa.String(length=255), nullable=True),
        sa.Column("account", sa.String(length=128), nullable=True),
        sa.Column("bank", sa.String(length=64), nullable=True),
        sa.Column("account_type", sa.String(length=64), nullable=True),
        sa.Column("extra", JSON, nullable=True),
        sa.ForeignKeyConstraint(
            ["transaction_id"],
            ["transactions.id"],
            name=op.f("fk_transaction_parties_transaction_id_transactions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_transaction_parties")),
        sa.UniqueConstraint(
            "transaction_id", "role", name=op.f("uq_transaction_parties_transaction_id_role")
        ),
    )
    op.create_index(op.f("ix_transaction_parties_phone"), "transaction_parties", ["phone"])
    op.create_index(op.f("ix_transaction_parties_account"), "transaction_parties", ["account"])


def downgrade() -> None:
    # Dropping a table drops its indexes; children before parents for the foreign keys.
    op.drop_table("transaction_parties")
    op.drop_table("transactions")
    op.drop_table("attachments")
    op.drop_table("messages")
    op.drop_table("users")
