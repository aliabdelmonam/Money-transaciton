"""split message parts: one row per WhatsApp message of a long reply

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-29 07:43:46.782665+00:00

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: Union[str, Sequence[str], None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Batch mode: SQLite can't add a foreign key in place, so the table is
    # rebuilt with its rows copied; other databases get plain ALTERs.
    with op.batch_alter_table("messages") as batch_op:
        batch_op.add_column(sa.Column("part_of_id", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("part_index", sa.Integer(), server_default="0", nullable=False))
        batch_op.create_index(batch_op.f("ix_messages_part_of_id"), ["part_of_id"])
        batch_op.create_foreign_key(
            batch_op.f("fk_messages_part_of_id_messages"),
            "messages",
            ["part_of_id"],
            ["id"],
            ondelete="CASCADE",
        )


def downgrade() -> None:
    with op.batch_alter_table("messages") as batch_op:
        batch_op.drop_constraint(batch_op.f("fk_messages_part_of_id_messages"), type_="foreignkey")
        batch_op.drop_index(batch_op.f("ix_messages_part_of_id"))
        batch_op.drop_column("part_index")
        batch_op.drop_column("part_of_id")
