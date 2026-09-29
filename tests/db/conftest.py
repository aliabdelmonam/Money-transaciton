from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config

from db import Database

ROOT = Path(__file__).resolve().parents[2]


def alembic_config(url: str) -> Config:
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", url)
    return config


@pytest.fixture
def make_alembic_config():
    return alembic_config


@pytest.fixture
def db_url(tmp_path: Path) -> str:
    return f"sqlite+aiosqlite:///{(tmp_path / 'test.db').as_posix()}"


@pytest.fixture
def migrated_url(db_url: str) -> str:
    """A fresh database built by the migrations, not by ``create_all``."""
    command.upgrade(alembic_config(db_url), "head")
    return db_url


@pytest.fixture
async def database(migrated_url: str):
    database = Database(migrated_url)
    yield database
    await database.dispose()
