"""Alembic environment: runs migrations against the app's async database."""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection, make_url
from sqlalchemy.ext.asyncio import create_async_engine

import db.models  # noqa: F401  registers every table on Base.metadata
from db.base import Base
from db.session import prepare_sqlite_path

config = context.config
if config.config_file_name is not None and config.attributes.get("configure_logger", True):
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def database_url() -> str:
    """``-x url=...`` first, then ``sqlalchemy.url`` if set, then the app settings."""
    url = context.get_x_argument(as_dictionary=True).get("url") or config.get_main_option("sqlalchemy.url")
    if url:
        return url
    from channels.config import Settings

    return Settings().database_url


def configure(database: str, **kwargs) -> None:
    context.configure(
        target_metadata=target_metadata,
        compare_type=True,
        # SQLite can't ALTER most things; batch mode recreates the table instead.
        render_as_batch=make_url(database).get_backend_name() == "sqlite",
        **kwargs,
    )


def run_migrations_offline() -> None:
    """Emit SQL to stdout (``alembic upgrade head --sql``) without connecting."""
    url = database_url()
    configure(url, url=url, literal_binds=True, dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection, url: str) -> None:
    configure(url, connection=connection)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    url = database_url()
    prepare_sqlite_path(url)
    engine = create_async_engine(url, poolclass=pool.NullPool)
    async with engine.connect() as connection:
        await connection.run_sync(do_run_migrations, url)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_async_migrations())
