"""transactions only: one flat table per receipt, drop users / messages / attachments / parties

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-29 16:00:00.000000+00:00

"""
import importlib.util
from pathlib import Path
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0003"
down_revision: Union[str, Sequence[str], None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

JSON = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
MONEY = sa.Numeric(precision=18, scale=2)


def party(side: str) -> list[sa.Column]:
    return [
        sa.Column(f"{side}_name", sa.String(length=255), nullable=True),
        sa.Column(f"{side}_phone", sa.String(length=32), nullable=True),
        sa.Column(f"{side}_email", sa.String(length=255), nullable=True),
        sa.Column(f"{side}_account", sa.String(length=128), nullable=True),
        sa.Column(f"{side}_bank", sa.String(length=64), nullable=True),
    ]


def upgrade() -> None:
    # Children before parents for the foreign keys. Their rows are not carried over:
    # the old transactions table was never filled, and messages are no longer kept.
    op.drop_table("transaction_parties")
    op.drop_table("transactions")
    op.drop_table("attachments")
    op.drop_table("messages")
    op.drop_table("users")

    op.create_table(
        "transactions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("channel", sa.String(length=32), nullable=False),
        sa.Column("message_id", sa.String(length=128), nullable=True),
        sa.Column("user_phone", sa.String(length=64), nullable=False),
        sa.Column("user_name", sa.String(length=255), nullable=True),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("caption", sa.Text(), nullable=True),
        sa.Column("image", sa.LargeBinary(), nullable=True),
        sa.Column("image_mime_type", sa.String(length=64), nullable=True),
        sa.Column("image_filename", sa.String(length=255), nullable=True),
        sa.Column("image_size", sa.Integer(), nullable=True),
        sa.Column("image_sha256", sa.String(length=64), nullable=True),
        sa.Column(
            "extraction_status",
            sa.Enum("pending", "extracted", "no_data", "failed", name="extractionstatus",
                    native_enum=False, create_constraint=False, length=16),
            nullable=False,
        ),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("provider", sa.String(length=64), nullable=True),
        sa.Column("transfer_status", sa.String(length=32), nullable=True),
        sa.Column("transfer_type", sa.String(length=64), nullable=True),
        sa.Column("amount", MONEY, nullable=True),
        sa.Column("fees", MONEY, nullable=True),
        sa.Column("total", MONEY, nullable=True),
        sa.Column("currency", sa.String(length=3), nullable=True),
        sa.Column("reference_id", sa.String(length=128), nullable=True),
        sa.Column("date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        *party("sender"),
        *party("receiver"),
        sa.Column("other_data", JSON, nullable=True),
        sa.Column("raw_result", JSON, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_transactions")),
        sa.UniqueConstraint("channel", "message_id", name=op.f("uq_transactions_channel_message_id")),
    )
    op.create_index(op.f("ix_transactions_user_phone"), "transactions", ["user_phone"])
    op.create_index(op.f("ix_transactions_image_sha256"), "transactions", ["image_sha256"])
    op.create_index(op.f("ix_transactions_extraction_status"), "transactions", ["extraction_status"])
    op.create_index(op.f("ix_transactions_date"), "transactions", ["date"])
    op.create_index(op.f("ix_transactions_sender_phone"), "transactions", ["sender_phone"])
    op.create_index(op.f("ix_transactions_receiver_phone"), "transactions", ["receiver_phone"])
    op.create_index("ix_transactions_provider_reference", "transactions", ["provider", "reference_id"])


def downgrade() -> None:
    # Back to the 0002 schema (empty): rebuild it with those revisions' own upgrade steps.
    op.drop_table("transactions")
    for name in ("20260929_0001_initial_schema.py", "20260929_0002_split_message_parts.py"):
        spec = importlib.util.spec_from_file_location(name[:-3], Path(__file__).with_name(name))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.upgrade()
