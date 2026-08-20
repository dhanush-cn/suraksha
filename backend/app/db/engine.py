"""Async SQLAlchemy engine + session factory.

Design:

* **One engine per process.** SQLAlchemy pools connections internally;
  creating a second engine defeats pooling and blows past the DB's
  ``max_connections``. This module holds the singleton behind
  :func:`get_engine`.
* **DSN normalisation.** :func:`_normalise_dsn` rewrites the naked
  driver names your ops team is likely to hand us -- ``postgres://``,
  ``postgresql://``, ``sqlite:///`` -- into the async-driver variants
  SQLAlchemy 2.0 requires. Prevents the "DSN works with psycopg2 but
  crashes with asyncpg" foot-gun.
* **SQLite quirk.** aiosqlite doesn't support concurrent transactions;
  the pool is forced to ``NullPool`` so each session gets a fresh
  connection. In production (postgres+asyncpg) the default
  ``AsyncAdaptedQueuePool`` is used, with pool sizes from Settings.
* **Testability.** :func:`session_scope` is an async context manager
  that yields a session and commits/rolls back around the block --
  the shape the repository layer and route handlers depend on.
"""

from __future__ import annotations

import contextlib
import logging
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _normalise_dsn(raw: str) -> str:
    """Rewrite common sync DSNs to their async-driver variants.

    ``postgres://`` and ``postgresql://`` -> ``postgresql+asyncpg://``
    ``sqlite:///``                          -> ``sqlite+aiosqlite:///``

    Leaves DSNs that already name a driver (``postgresql+asyncpg://``,
    ``sqlite+aiosqlite:///``, ``mysql+aiomysql://``) untouched.
    """
    if raw.startswith("postgres://") and "+" not in raw.split("://", 1)[0]:
        # Heroku, Railway, some managed-Postgres URLs.
        return "postgresql+asyncpg://" + raw.split("://", 1)[1]
    if raw.startswith("postgresql://") and "+" not in raw.split("://", 1)[0]:
        return "postgresql+asyncpg://" + raw.split("://", 1)[1]
    if raw.startswith("sqlite://") and "+" not in raw.split("://", 1)[0]:
        return "sqlite+aiosqlite://" + raw.split("://", 1)[1]
    return raw


def _build_engine() -> AsyncEngine:
    settings = get_settings()
    dsn = _normalise_dsn(settings.database_url.get_secret_value())

    kwargs: dict = {
        "echo": settings.database_echo,
        "future": True,
    }
    if dsn.startswith("sqlite"):
        # aiosqlite serialises all writes anyway; pooling multiple
        # connections just wastes file descriptors and can trip
        # "database is locked" under concurrent writes.
        kwargs["poolclass"] = NullPool
        # aiosqlite requires check_same_thread=False for the async
        # driver's thread model.
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        kwargs["pool_size"] = settings.database_pool_size
        kwargs["max_overflow"] = settings.database_max_overflow
        # Recycle sockets before typical PgBouncer / cloud proxy
        # idle-close windows (usually 5-10 min). 1800s is conservative.
        kwargs["pool_pre_ping"] = True
        kwargs["pool_recycle"] = 1800

    logger.info("creating async engine for %s", dsn.split("://", 1)[0])
    return create_async_engine(dsn, **kwargs)


def get_engine() -> AsyncEngine:
    """Return the process-wide async engine, building it on first call."""
    global _engine, _session_factory
    if _engine is None:
        _engine = _build_engine()
        _session_factory = async_sessionmaker(
            _engine,
            # expire_on_commit=False: keep the mapped attributes usable
            # after commit() without triggering a lazy-load round trip.
            # In async code, an accidental lazy-load raises MissingGreenlet
            # rather than just being slow, so this default is safer.
            expire_on_commit=False,
            class_=AsyncSession,
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    if _session_factory is None:
        get_engine()
    assert _session_factory is not None
    return _session_factory


@contextlib.asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Transactional scope around a series of operations.

    Usage::

        async with session_scope() as session:
            user = await session.get(User, 1)
            user.last_login_at = datetime.now(timezone.utc)
            # commit happens automatically on clean exit

    On any exception the transaction is rolled back before the exception
    re-raises -- there's no "half-written" state left in the DB.
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def dispose_engine() -> None:
    """Close the pool. Call from FastAPI shutdown or test teardown."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None


def reset_engine_for_tests() -> None:
    """Drop the cached engine WITHOUT awaiting close.

    Only for tests that swap DATABASE_URL between runs and can't afford
    an ``await`` (module-level pytest fixtures). Production code should
    always use :func:`dispose_engine`.
    """
    global _engine, _session_factory
    _engine = None
    _session_factory = None


__all__ = [
    "dispose_engine",
    "get_engine",
    "get_session_factory",
    "reset_engine_for_tests",
    "session_scope",
]
