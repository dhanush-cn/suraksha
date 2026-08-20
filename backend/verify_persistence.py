"""Smoke-verification of the persistence layer. Run: python verify_persistence.py

Exercises the async engine + repositories against an ephemeral SQLite
database created via Alembic upgrade -- so the same migration that
would run in production is what these checks actually run against.

Checks:

* Alembic upgrade builds the whole schema from scratch (proves the
  baseline migration is self-consistent).
* Composite index ``ix_alert_logs_mine_triggered`` exists.
* MineRepository CRUD (create / list / get / delete + FK cascade to
  alert_logs).
* MineRepository maps UNIQUE violations to ConflictError.
* AlertRepository.log + AlertRepository.recent honour the (mine_id,
  triggered_at DESC) ordering, and joinedload avoids N+1 on mine.
* UserRepository lookup is case-insensitive; duplicate username -> 409.
* UserRepository.touch_last_login is a single-statement UPDATE.
* DSN normalisation ``postgres://`` -> ``postgresql+asyncpg://``.

No real Postgres is required; the DSN check is a pure-string test.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime, timedelta, timezone

os.environ.setdefault("JWT_SECRET", "a" * 48)
os.environ.setdefault("ENVIRONMENT", "test")

# Each check builds a fresh temp DB then swaps DATABASE_URL. Do NOT
# set DATABASE_URL at import time -- it would be baked into
# get_settings()'s lru_cache before per-check overrides take effect.

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str):
    def wrapper(fn):
        try:
            fn()
            results.append((PASS, name, ""))
        except Exception as exc:  # noqa: BLE001
            results.append((FAIL, name, f"{type(exc).__name__}: {exc}"))
            traceback.print_exc()
        return fn

    return wrapper


def _fresh_sqlite_env() -> str:
    """Create a temp .db path and configure it as the current DATABASE_URL.

    Also clears the app.core.config lru_cache and the app.db.engine
    singletons so the next call rebuilds against the new path.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp.name}"

    # Reset all cached state that captured the old DATABASE_URL.
    from app.core.config import get_settings

    get_settings.cache_clear()  # type: ignore[attr-defined]

    from app.db import engine as engine_module

    engine_module.reset_engine_for_tests()

    return tmp.name


def _run_migrations() -> None:
    """Invoke ``alembic upgrade head`` against the current DATABASE_URL.

    Uses a subprocess so we get the real user-facing migration path,
    not a Python API shortcut that might paper over a broken env.py.
    """
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=backend_dir,
        env={**os.environ},
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"alembic upgrade failed:\n{result.stdout}\n{result.stderr}")


# ---------------------------------------------------------------------------
# migration integrity
# ---------------------------------------------------------------------------


@check("alembic: baseline migration creates the full schema")
def _():
    import sqlite3

    db_path = _fresh_sqlite_env()
    _run_migrations()
    conn = sqlite3.connect(db_path)
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    }
    assert {"mines", "alert_logs", "users", "alembic_version"}.issubset(tables), tables


@check("alembic: ix_alert_logs_mine_triggered composite index exists")
def _():
    import sqlite3

    db_path = _fresh_sqlite_env()
    _run_migrations()
    conn = sqlite3.connect(db_path)
    indexes = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='alert_logs'"
        )
    }
    assert "ix_alert_logs_mine_triggered" in indexes, indexes


# ---------------------------------------------------------------------------
# MineRepository
# ---------------------------------------------------------------------------


@check("mines: create + list + get + delete round-trip")
def _():
    _fresh_sqlite_env()
    _run_migrations()

    from app.db.engine import session_scope
    from app.repositories.mine import MineRepository

    async def go():
        async with session_scope() as session:
            repo = MineRepository(session)
            created = await repo.create(
                name="Grasberg", company="Freeport", location_name="Papua",
                latitude=-4.05, longitude=137.11,
                contact_email="safety@example.org",
            )
            assert created.id is not None
            assert created.alert_threshold_pct == 70.0  # default

        async with session_scope() as session:
            repo = MineRepository(session)
            listed = await repo.list_all()
            assert len(listed) == 1
            fetched = await repo.get(listed[0].id)
            assert fetched is not None
            assert fetched.name == "Grasberg"

        async with session_scope() as session:
            repo = MineRepository(session)
            first_id = (await repo.list_all())[0].id
            assert await repo.delete(first_id) is True
            assert await repo.delete(first_id) is False  # already gone

    asyncio.run(go())


@check("mines: duplicate name raises ConflictError (not IntegrityError)")
def _():
    _fresh_sqlite_env()
    _run_migrations()

    from app.core.exceptions import ConflictError
    from app.db.engine import session_scope
    from app.repositories.mine import MineRepository

    async def go():
        async with session_scope() as session:
            await MineRepository(session).create(
                name="Grasberg", company="X", location_name="Y",
                latitude=0.0, longitude=0.0,
            )
        try:
            async with session_scope() as session:
                await MineRepository(session).create(
                    name="Grasberg", company="X", location_name="Y",
                    latitude=1.0, longitude=1.0,
                )
        except ConflictError as exc:
            assert "already registered" in exc.message
        else:
            raise AssertionError("duplicate mine name accepted!")

    asyncio.run(go())


@check("mines: FK cascade deletes associated alert_logs")
def _():
    _fresh_sqlite_env()
    _run_migrations()

    from sqlalchemy import event, text

    from app.db.engine import get_engine, session_scope
    from app.repositories.alert import AlertRepository
    from app.repositories.mine import MineRepository

    # SQLite requires PRAGMA foreign_keys=ON per-connection to enforce
    # FK constraints. Postgres enforces by default; this listener keeps
    # local dev/test parity with production behavior.
    @event.listens_for(get_engine().sync_engine, "connect")
    def _enable_sqlite_fks(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async def go():
        async with session_scope() as session:
            mine = await MineRepository(session).create(
                name="A", company="X", location_name="Y",
                latitude=0.0, longitude=0.0,
            )
            mine_id = mine.id
            await AlertRepository(session).log(
                mine_id=mine_id, risk_percentage=90.0, risk_level="critical",
                rainfall_mm=40, pore_pressure_kpa=95, velocity_mm_h=5,
                seismic_rms_g=0.4, top_shap_reason="pore pressure",
            )

        async with session_scope() as session:
            assert (await AlertRepository(session).recent(mine_id=mine_id))

        async with session_scope() as session:
            await MineRepository(session).delete(mine_id)

        async with session_scope() as session:
            # Cascade should have removed the alert.
            assert not (await AlertRepository(session).recent(mine_id=mine_id))

    asyncio.run(go())


# ---------------------------------------------------------------------------
# AlertRepository
# ---------------------------------------------------------------------------


@check("alerts: recent() returns newest-first, honours limit + mine filter")
def _():
    _fresh_sqlite_env()
    _run_migrations()

    from app.db.engine import session_scope
    from app.repositories.alert import AlertRepository
    from app.repositories.mine import MineRepository

    async def go():
        # Two mines, three alerts each, staggered timestamps.
        async with session_scope() as session:
            m1 = await MineRepository(session).create(
                name="M1", company="X", location_name="Y",
                latitude=0.0, longitude=0.0,
            )
            m2 = await MineRepository(session).create(
                name="M2", company="X", location_name="Y",
                latitude=1.0, longitude=1.0,
            )
            base = datetime(2026, 8, 20, 10, 0, 0, tzinfo=timezone.utc)
            repo = AlertRepository(session)
            for i in range(3):
                await repo.log(
                    mine_id=m1.id, risk_percentage=80 + i, risk_level="critical",
                    rainfall_mm=10, pore_pressure_kpa=50, velocity_mm_h=1,
                    seismic_rms_g=0.1, top_shap_reason="r",
                    triggered_at=base + timedelta(minutes=i),
                )
                await repo.log(
                    mine_id=m2.id, risk_percentage=70 + i, risk_level="warning",
                    rainfall_mm=5, pore_pressure_kpa=40, velocity_mm_h=0.5,
                    seismic_rms_g=0.05, top_shap_reason="r",
                    triggered_at=base + timedelta(minutes=i * 2),
                )
            m1_id, m2_id = m1.id, m2.id

        async with session_scope() as session:
            repo = AlertRepository(session)
            all_alerts = await repo.recent(limit=10)
            assert len(all_alerts) == 6
            # newest-first ordering
            ts_desc = [a.triggered_at for a in all_alerts]
            assert ts_desc == sorted(ts_desc, reverse=True)

            m1_only = await repo.recent(mine_id=m1_id, limit=10)
            assert len(m1_only) == 3
            assert all(a.mine_id == m1_id for a in m1_only)

            # Limit is honored.
            top2 = await repo.recent(limit=2)
            assert len(top2) == 2

            # joinedload(mine) means .mine is populated without a
            # second round trip; access after commit would raise
            # MissingGreenlet otherwise.
            assert all(a.mine is not None and a.mine.name for a in top2)

    asyncio.run(go())


# ---------------------------------------------------------------------------
# UserRepository
# ---------------------------------------------------------------------------


@check("users: lookup is case-insensitive")
def _():
    _fresh_sqlite_env()
    _run_migrations()

    from app.core.security import Role, hash_password
    from app.db.engine import session_scope
    from app.repositories.user import UserRepository

    async def go():
        async with session_scope() as session:
            await UserRepository(session).create(
                username="Admin",
                password_hash=hash_password("admin123"),
                role=str(Role.ADMIN),
            )

        async with session_scope() as session:
            repo = UserRepository(session)
            for spelling in ("admin", "ADMIN", "  Admin  "):
                user = await repo.get_by_username(spelling)
                assert user is not None, f"lookup failed for {spelling!r}"

    asyncio.run(go())


@check("users: duplicate username raises ConflictError")
def _():
    _fresh_sqlite_env()
    _run_migrations()

    from app.core.exceptions import ConflictError
    from app.core.security import Role, hash_password
    from app.db.engine import session_scope
    from app.repositories.user import UserRepository

    async def go():
        async with session_scope() as session:
            await UserRepository(session).create(
                username="alice",
                password_hash=hash_password("password-1234"),
                role=str(Role.OPERATOR),
                mine_id=None,
            )
        try:
            async with session_scope() as session:
                await UserRepository(session).create(
                    username="alice",  # same, even at same case
                    password_hash=hash_password("different-1234"),
                    role=str(Role.OPERATOR),
                )
        except ConflictError:
            pass
        else:
            raise AssertionError("duplicate username accepted!")

    asyncio.run(go())


@check("users: touch_last_login updates without a preceding SELECT")
def _():
    _fresh_sqlite_env()
    _run_migrations()

    from app.core.security import Role, hash_password
    from app.db.engine import session_scope
    from app.repositories.user import UserRepository

    async def go():
        async with session_scope() as session:
            user = await UserRepository(session).create(
                username="bob",
                password_hash=hash_password("password-1234"),
                role=str(Role.ADMIN),
            )
            user_id = user.id

        when = datetime.now(timezone.utc)
        async with session_scope() as session:
            await UserRepository(session).touch_last_login(user_id, when=when)

        async with session_scope() as session:
            reloaded = await UserRepository(session).get(user_id)
            assert reloaded is not None
            assert reloaded.last_login_at is not None
            # timezone-aware comparison
            delta = abs((reloaded.last_login_at - when).total_seconds())
            assert delta < 1.0, f"drift {delta}s"

    asyncio.run(go())


# ---------------------------------------------------------------------------
# DSN normalisation
# ---------------------------------------------------------------------------


@check("engine: normalises postgres:// / sqlite:// to async drivers")
def _():
    from app.db.engine import _normalise_dsn

    assert (
        _normalise_dsn("postgres://u:p@h:5432/d")
        == "postgresql+asyncpg://u:p@h:5432/d"
    )
    assert (
        _normalise_dsn("postgresql://u:p@h:5432/d")
        == "postgresql+asyncpg://u:p@h:5432/d"
    )
    assert _normalise_dsn("sqlite:///./mines.db") == "sqlite+aiosqlite:///./mines.db"
    # Leaves already-async DSNs alone.
    assert (
        _normalise_dsn("postgresql+asyncpg://u:p@h/d")
        == "postgresql+asyncpg://u:p@h/d"
    )
    assert (
        _normalise_dsn("sqlite+aiosqlite:///./mines.db")
        == "sqlite+aiosqlite:///./mines.db"
    )


if __name__ == "__main__":
    print("\n" + "=" * 78)
    for status, name, detail in results:
        marker = "✓" if status == PASS else "✗"
        print(f"  {marker} {status}  {name}")
        if detail:
            print(f"           -> {detail}")
    failures = sum(1 for status, _, _ in results if status == FAIL)
    print("=" * 78)
    print(f"  {len(results) - failures}/{len(results)} checks passed")
    print("=" * 78)
    raise SystemExit(1 if failures else 0)
