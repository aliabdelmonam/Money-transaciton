"""Declarative base and column types shared by every table."""

from datetime import datetime
from enum import Enum as PyEnum
from typing import Any

from sqlalchemy import JSON, DateTime, Enum, MetaData, Numeric, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Deterministic constraint names, so migrations can drop/alter them later
# (SQLite batch mode cannot touch an unnamed constraint).
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

# JSON everywhere, JSONB on Postgres (indexable, compact).
JSONType = JSON().with_variant(JSONB(), "postgresql")

# Two decimal places covers EGP/USD/SAR... amounts; never store money as float.
Money = Numeric(18, 2)


def str_enum(enum_cls: type[PyEnum], length: int = 16) -> Enum:
    """Store a ``str`` enum by value in a VARCHAR.

    No native DB enum and no CHECK constraint, so adding a member later
    needs no migration.
    """
    return Enum(
        enum_cls,
        native_enum=False,
        create_constraint=False,
        length=length,
        values_callable=lambda members: [member.value for member in members],
    )


class Base(AsyncAttrs, DeclarativeBase):
    """``await obj.awaitable_attrs.<name>`` loads a lazy/deferred attribute in async code."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)
    type_annotation_map = {dict[str, Any]: JSONType}


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
