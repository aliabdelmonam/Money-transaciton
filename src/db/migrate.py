"""Apply Alembic migrations from code, e.g. at app startup."""

from pathlib import Path

from alembic import command
from alembic.config import Config

ALEMBIC_INI = Path(__file__).resolve().parents[2] / "alembic.ini"


def upgrade(url: str, revision: str = "head") -> None:
    """Migrate ``url`` to ``revision``.

    Blocking, and env.py starts its own event loop: from async code call it
    with ``await asyncio.to_thread(upgrade, url)``.
    """
    # Leave the app's logging alone; alembic.ini's [loggers] are for the CLI.
    config = Config(str(ALEMBIC_INI), attributes={"configure_logger": False})
    config.set_main_option("sqlalchemy.url", url.replace("%", "%%"))  # ini interpolation
    command.upgrade(config, revision)
