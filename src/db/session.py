"""Async engine and session factory."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from sqlalchemy import event
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine


def prepare_sqlite_path(url: str) -> None:
    """Create the folder of a file-based SQLite database (e.g. ``./data``)."""
    parsed = make_url(url)
    if parsed.get_backend_name() == "sqlite" and parsed.database not in (None, "", ":memory:"):
        Path(parsed.database).parent.mkdir(parents=True, exist_ok=True)


def enable_sqlite_foreign_keys(engine: Engine) -> None:
    """SQLite ignores foreign keys (and ON DELETE CASCADE) unless asked per connection."""

    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_connection, _record) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


class Database:
    """Owns the engine; hand out sessions with :meth:`session`."""

    def __init__(self, url: str, echo: bool = False):
        prepare_sqlite_path(url)
        self.engine = create_async_engine(url, echo=echo)
        if self.engine.dialect.name == "sqlite":
            enable_sqlite_foreign_keys(self.engine.sync_engine)
        self.sessionmaker = async_sessionmaker(self.engine, expire_on_commit=False)

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """One unit of work: committed on success, rolled back on error."""
        async with self.sessionmaker() as session:
            try:
                yield session
                await session.commit()
            except BaseException:
                await session.rollback()
                raise

    async def dispose(self) -> None:
        await self.engine.dispose()
