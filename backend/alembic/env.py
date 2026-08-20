"""Alembic environment.

Both online (against a live async engine) and offline (SQL script
generation) modes are supported. In online mode we reach into
:func:`app.db.engine.get_engine` so the same DSN normalisation applies
that the runtime uses -- avoids the classic "migration works, app
crashes" split-brain when someone sets ``postgres://`` in one place and
``postgresql+asyncpg://`` in the other.

Autogenerate diffs against :data:`app.db.models.Base.metadata`. All ORM
models must be imported at module import time; if a new model is added
somewhere else, import it here so Alembic sees it.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy.ext.asyncio import AsyncConnection

# Make the backend/ directory importable regardless of where alembic
# was launched from. When run as ``alembic -c backend/alembic.ini ...``
# from the repo root, sys.path only includes the CWD; we need
# ``backend/`` on it so ``import app.*`` succeeds.
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Match main.py's defaults so `alembic upgrade head` from a bare
# checkout doesn't require exporting env vars.
os.environ.setdefault(
    "JWT_SECRET",
    "rockfallguard-dev-only-insecure-jwt-secret-do-not-use-in-prod",
)
os.environ.setdefault("DATABASE_URL", "sqlite:///./mines.db")

from app.db.base import Base  # noqa: E402
from app.db.engine import _normalise_dsn, get_engine  # noqa: E402

# Ensure every model is loaded before autogenerate reads metadata.
import app.db.models  # noqa: E402,F401

logger = logging.getLogger("alembic.env")

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _url_for_context() -> str:
    """DSN Alembic should stamp into migration output.

    We hand it a *sync* driver name so ``alembic upgrade`` invoked
    without a live app context works too (Alembic's own sqlalchemy
    session is sync). :func:`get_engine` runs its own async engine on
    the normalised URL for actual online migrations.
    """
    from app.core.config import get_settings

    raw = get_settings().database_url.get_secret_value()
    # Alembic offline mode wants a *sync* URL, so strip the async
    # driver hint if present.
    stripped = raw.replace("+asyncpg", "").replace("+aiosqlite", "")
    return stripped


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting."""
    context.configure(
        url=_url_for_context(),
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
        # dialect_opts={"paramstyle": "named"} would kick in if we ever
        # generate parameterised SQL; not needed with literal_binds.
    )
    with context.begin_transaction():
        context.run_migrations()


def _run_sync_migrations(connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        # SQLite lacks ALTER TABLE for many operations, so future
        # migrations that touch existing columns will need
        # render_as_batch=True. Kept off by default because it changes
        # emitted SQL cosmetically and shouldn't fire pre-emptively.
    )
    context.run_migrations()


async def run_migrations_online() -> None:
    """Run migrations against a live async engine."""
    engine = get_engine()
    async with engine.begin() as conn:
        assert isinstance(conn, AsyncConnection)
        await conn.run_sync(_run_sync_migrations)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    # asyncio.run because Alembic doesn't run in an event loop by default.
    asyncio.run(run_migrations_online())
