import io
from contextlib import redirect_stdout

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, inspect

import db.models  # noqa: F401
from db.base import Base

TABLES = {"transactions"}


def sync_url(url: str) -> str:
    return url.replace("sqlite+aiosqlite", "sqlite")


def test_migrations_match_models(migrated_url):
    """Fails when a model changes without a new migration (run ``alembic revision --autogenerate``)."""
    engine = create_engine(sync_url(migrated_url))
    with engine.connect() as connection:
        context = MigrationContext.configure(connection, opts={"compare_type": True})
        diff = compare_metadata(context, Base.metadata)
    engine.dispose()
    assert diff == []


def test_downgrade_then_upgrade(migrated_url, make_alembic_config):
    config = make_alembic_config(migrated_url)
    engine = create_engine(sync_url(migrated_url))

    command.downgrade(config, "base")
    assert set(inspect(engine).get_table_names()) == {"alembic_version"}

    command.upgrade(config, "head")
    assert set(inspect(engine).get_table_names()) == TABLES | {"alembic_version"}
    engine.dispose()


def test_offline_sql_for_postgres(make_alembic_config):
    out = io.StringIO()
    with redirect_stdout(out):
        command.upgrade(make_alembic_config("postgresql://user:pass@localhost/money"), "head", sql=True)
    sql = out.getvalue()
    assert "CREATE TABLE transactions" in sql
    assert "JSONB" in sql
    assert "NUMERIC(18, 2)" in sql
