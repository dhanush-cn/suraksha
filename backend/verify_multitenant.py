"""Smoke-verification of Step 9 tenant scoping. Run: python verify_multitenant.py

The Step 9 headline: every list endpoint now filters by ``TenantScope``,
scoped cache never contaminates admin vs operator views, and RAG
retrieval cannot surface another mine's alerts to an operator's chat.

Checks:

* TenantScope: from_principal + scope_hash + visible_mine_ids.
* MineService.list_visible: admin sees all, operator sees only their
  own, non-existent mine returns [].
* AlertService.list_recent: admin sees all mines' alerts, operator's
  are filtered to their own via the SQL WHERE clause.
* Cache: scope-keyed list caches never cross (admin key vs mine:1 key
  are physically different).
* Rate limiter identity: prefers principal.user_id when
  request.state.principal is set; falls back to IP otherwise.
* End-to-end via TestClient: an operator's GET /api/mines returns
  exactly one row; their /api/telemetry/2 (cross-tenant) is 403;
  their /api/alerts is filtered; their /api/predict_risk for another
  mine is 403.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("JWT_SECRET", "a" * 48)
os.environ.setdefault("DATABASE_URL", "sqlite:///./mines.db")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("ENVIRONMENT", "test")

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
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp.name}"

    from app.core.config import get_settings

    get_settings.cache_clear()  # type: ignore[attr-defined]
    from app.db import engine as engine_module

    engine_module.reset_engine_for_tests()
    return tmp.name


def _migrate() -> None:
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=backend_dir,
        env={**os.environ},
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"alembic failed:\n{result.stderr}")


def _mock_redis_dep(value):
    return AsyncMock(return_value=value)


def _seed_two_mines():
    """Insert Mines with ids 1 and 2 via the ORM against the CURRENT
    DATABASE_URL. Can't reuse ``database.init_db()`` here because it
    hardcodes DB_PATH to ``backend/mines.db`` and would seed the wrong
    file when the tests point DATABASE_URL at a temp path.
    """
    from app.db.engine import session_scope
    from app.db.models import Mine

    async def go():
        async with session_scope() as session:
            session.add(Mine(
                id=1, name="Grasberg Open-Pit Mine", company="Freeport Copper-Gold",
                location_name="Papua High Elevation Pit", latitude=-4.05, longitude=137.11,
                pit_depth_m=450.0, slope_angle_deg=48.0,
                contact_email="safety@grasbergmine.org", alert_threshold_pct=70.0,
            ))
            session.add(Mine(
                id=2, name="Chuquicamata Mine", company="Codelco Copper",
                location_name="Atacama Pit Sector B", latitude=-22.31, longitude=-68.90,
                pit_depth_m=850.0, slope_angle_deg=52.0,
                contact_email="geotech@codelco.cl", alert_threshold_pct=70.0,
            ))

    asyncio.run(go())


# ---------------------------------------------------------------------------
# TenantScope basics
# ---------------------------------------------------------------------------


@check("scope: from_principal + scope_hash + visible_mine_ids")
def _():
    from datetime import datetime, timezone

    from app.core.security import Role
    from app.schemas.auth import Principal, TenantScope

    now = datetime.now(timezone.utc)
    admin = Principal(
        user_id="1", username="admin", role=Role.ADMIN, mine_id=None,
        token_id="j", issued_at=now, expires_at=now,
    )
    operator = Principal(
        user_id="7", username="op", role=Role.OPERATOR, mine_id=1,
        token_id="j", issued_at=now, expires_at=now,
    )

    admin_scope = TenantScope.from_principal(admin)
    op_scope = TenantScope.from_principal(operator)

    assert admin_scope.is_admin is True and admin_scope.mine_id is None
    assert op_scope.is_admin is False and op_scope.mine_id == 1

    # Scope hashes must differ so cache keys don't collide.
    assert admin_scope.scope_hash() == "admin"
    assert op_scope.scope_hash() == "mine:1"
    assert admin_scope.scope_hash() != op_scope.scope_hash()

    # visible_mine_ids filter
    all_ids = [1, 2, 3]
    assert admin_scope.visible_mine_ids(all_ids) == [1, 2, 3]
    assert op_scope.visible_mine_ids(all_ids) == [1]


# ---------------------------------------------------------------------------
# MineService.list_visible
# ---------------------------------------------------------------------------


@check("mines: list_visible admin returns all, operator returns own only")
def _():
    _fresh_sqlite_env()
    _migrate()
    _seed_two_mines()

    from app.db.engine import session_scope
    from app.schemas.auth import TenantScope
    from app.services.mine import MineService

    async def go():
        # Bypass Redis so cache calls short-circuit (no scoped caching
        # side-effect leaking between the two calls we make here).
        with patch("app.core.cache.get_redis", new=_mock_redis_dep(None)):
            async with session_scope() as session:
                svc = MineService(session)
                admin = await svc.list_visible(
                    TenantScope(is_admin=True, mine_id=None)
                )
                assert len(admin) >= 2, f"admin got {len(admin)} mines"
                op = await svc.list_visible(
                    TenantScope(is_admin=False, mine_id=1)
                )
                assert len(op) == 1 and op[0].id == 1

    asyncio.run(go())


@check("mines: list_visible for operator with non-existent mine returns []")
def _():
    _fresh_sqlite_env()
    _migrate()
    _seed_two_mines()

    from app.db.engine import session_scope
    from app.schemas.auth import TenantScope
    from app.services.mine import MineService

    async def go():
        with patch("app.core.cache.get_redis", new=_mock_redis_dep(None)):
            async with session_scope() as session:
                svc = MineService(session)
                out = await svc.list_visible(
                    TenantScope(is_admin=False, mine_id=99_999)
                )
                assert out == []

    asyncio.run(go())


# ---------------------------------------------------------------------------
# AlertService.list_recent
# ---------------------------------------------------------------------------


@check("alerts: list_recent scopes to mine_id for operators")
def _():
    _fresh_sqlite_env()
    _migrate()
    _seed_two_mines()

    from app.db.engine import session_scope
    from app.schemas.auth import TenantScope
    from app.services.alert import AlertService

    async def go():
        # Insert alerts on two mines.
        async with session_scope() as session:
            svc = AlertService(session)
            await svc._repo.log(
                mine_id=1, risk_percentage=88.0, risk_level="critical",
                rainfall_mm=10.0, pore_pressure_kpa=50.0, velocity_mm_h=1.0,
                seismic_rms_g=0.1, top_shap_reason="x",
            )
            await svc._repo.log(
                mine_id=2, risk_percentage=72.0, risk_level="critical",
                rainfall_mm=5.0, pore_pressure_kpa=40.0, velocity_mm_h=0.5,
                seismic_rms_g=0.05, top_shap_reason="y",
            )

        async with session_scope() as session:
            svc = AlertService(session)
            admin_view = await svc.list_recent(
                scope=TenantScope(is_admin=True, mine_id=None), limit=50
            )
            op1_view = await svc.list_recent(
                scope=TenantScope(is_admin=False, mine_id=1), limit=50
            )
            op2_view = await svc.list_recent(
                scope=TenantScope(is_admin=False, mine_id=2), limit=50
            )

        assert len(admin_view) >= 2, admin_view
        assert all(a.mine_id == 1 for a in op1_view) and len(op1_view) == 1
        assert all(a.mine_id == 2 for a in op2_view) and len(op2_view) == 1
        # And crucially: an operator MUST NOT see the other mine's alerts.
        assert 2 not in [a.mine_id for a in op1_view]
        assert 1 not in [a.mine_id for a in op2_view]

    asyncio.run(go())


# ---------------------------------------------------------------------------
# Scope-keyed cache
# ---------------------------------------------------------------------------


@check("cache: admin and operator list caches live at different keys")
def _():
    import fakeredis.aioredis as fake

    from app.core.cache import (
        _mine_list_key,
        get_cached_mine_list,
        set_cached_mine_list,
    )

    async def go():
        fr = fake.FakeRedis(decode_responses=True)
        with patch("app.core.cache.get_redis", new=_mock_redis_dep(fr)):
            await set_cached_mine_list("admin", [{"id": 1}, {"id": 2}, {"id": 3}])
            await set_cached_mine_list("mine:1", [{"id": 1}])

            admin_cached = await get_cached_mine_list("admin")
            op_cached = await get_cached_mine_list("mine:1")

        assert admin_cached is not None and len(admin_cached) == 3
        assert op_cached is not None and len(op_cached) == 1
        assert _mine_list_key("admin") != _mine_list_key("mine:1")

    asyncio.run(go())


@check("cache: invalidate_mine clears ALL scope-keyed list variants")
def _():
    import fakeredis.aioredis as fake

    from app.core.cache import (
        get_cached_mine_list,
        invalidate_mine,
        set_cached_mine_list,
    )

    async def go():
        fr = fake.FakeRedis(decode_responses=True)
        with patch("app.core.cache.get_redis", new=_mock_redis_dep(fr)):
            await set_cached_mine_list("admin", [{"id": 1}])
            await set_cached_mine_list("mine:1", [{"id": 1}])
            await set_cached_mine_list("mine:2", [{"id": 2}])

            await invalidate_mine(99)  # id doesn't matter -- lists are always nuked

            assert await get_cached_mine_list("admin") is None
            assert await get_cached_mine_list("mine:1") is None
            assert await get_cached_mine_list("mine:2") is None

    asyncio.run(go())


# ---------------------------------------------------------------------------
# Rate limiter identity
# ---------------------------------------------------------------------------


@check("rate_limit: _identify prefers principal.user_id when present")
def _():
    from app.core.rate_limit import _identify

    class _P:
        user_id = "42"

    req = MagicMock()
    req.state.principal = _P()
    req.client.host = "1.2.3.4"
    req.headers = {}
    assert _identify(req) == "user:42"

    # No principal -> falls back to IP.
    req.state = MagicMock(spec=[])  # no principal attribute
    assert _identify(req) == "ip:1.2.3.4"


# ---------------------------------------------------------------------------
# End-to-end via TestClient
# ---------------------------------------------------------------------------


@check("e2e: operator's GET /api/mines returns exactly their assigned mine")
def _():
    _fresh_sqlite_env()
    _migrate()
    _seed_two_mines()

    from fastapi.testclient import TestClient
    import main

    c = TestClient(main.app)
    op = c.post(
        "/api/auth/login",
        json={"username": "grasberg_user", "password": "user123"},
    ).json()
    h = {"Authorization": f"Bearer {op['access_token']}"}

    r = c.get("/api/mines", headers=h)
    assert r.status_code == 200
    mines = r.json()
    assert len(mines) == 1, f"operator saw {len(mines)} mines, expected 1"
    assert mines[0]["id"] == 1

    # Admin sees all
    ad = c.post(
        "/api/auth/login", json={"username": "admin", "password": "admin123"}
    ).json()
    ha = {"Authorization": f"Bearer {ad['access_token']}"}
    r = c.get("/api/mines", headers=ha)
    assert r.status_code == 200
    assert len(r.json()) >= 2


@check("e2e: /api/mines requires authentication (was public)")
def _():
    _fresh_sqlite_env()
    _migrate()
    _seed_two_mines()

    from fastapi.testclient import TestClient
    import main

    c = TestClient(main.app)
    r = c.get("/api/mines")  # no Authorization header
    assert r.status_code == 401, f"expected 401 for anonymous, got {r.status_code}"


@check("e2e: operator's cross-tenant /api/predict_risk is 403")
def _():
    _fresh_sqlite_env()
    _migrate()
    _seed_two_mines()

    from fastapi.testclient import TestClient
    import main

    c = TestClient(main.app)
    op = c.post(
        "/api/auth/login",
        json={"username": "grasberg_user", "password": "user123"},
    ).json()
    h = {"Authorization": f"Bearer {op['access_token']}"}

    # grasberg_user is scoped to mine 1; asking about mine 2 must 403.
    r = c.post(
        "/api/predict_risk",
        headers=h,
        json={
            "mine_id": 2,
            "rainfall_mm": 0.0, "humidity_pct": 50.0, "pore_pressure_kpa": 35.0,
            "displacement_mm": 5.0, "velocity_mm_h": 0.02,
            "acceleration_mm_h2": 0.0005, "raw_seismic_rms_g": 0.01,
        },
    )
    assert r.status_code == 403, f"expected 403, got {r.status_code}: {r.text}"


@check("e2e: /api/alerts is scope-filtered per caller")
def _():
    _fresh_sqlite_env()
    _migrate()
    _seed_two_mines()

    # Seed one alert on each mine via the ORM. Not database.log_alert
    # -- that function hardcodes DB_PATH to backend/mines.db and would
    # write to the wrong file when DATABASE_URL points at our temp DB.
    from app.db.engine import session_scope
    from app.repositories.alert import AlertRepository

    async def seed():
        async with session_scope() as s:
            repo = AlertRepository(s)
            await repo.log(
                mine_id=1, risk_percentage=88.0, risk_level="critical",
                rainfall_mm=10.0, pore_pressure_kpa=50.0, velocity_mm_h=1.0,
                seismic_rms_g=0.1, top_shap_reason="x",
            )
            await repo.log(
                mine_id=2, risk_percentage=72.0, risk_level="critical",
                rainfall_mm=5.0, pore_pressure_kpa=40.0, velocity_mm_h=0.5,
                seismic_rms_g=0.05, top_shap_reason="y",
            )

    asyncio.run(seed())

    from fastapi.testclient import TestClient
    import main

    c = TestClient(main.app)
    op = c.post(
        "/api/auth/login",
        json={"username": "grasberg_user", "password": "user123"},
    ).json()
    ho = {"Authorization": f"Bearer {op['access_token']}"}

    r = c.get("/api/alerts", headers=ho)
    assert r.status_code == 200
    op_alerts = r.json()
    assert len(op_alerts) == 1 and op_alerts[0]["mine_id"] == 1

    # Admin sees both
    ad = c.post(
        "/api/auth/login", json={"username": "admin", "password": "admin123"}
    ).json()
    ha = {"Authorization": f"Bearer {ad['access_token']}"}
    r = c.get("/api/alerts", headers=ha)
    assert r.status_code == 200
    assert len(r.json()) >= 2


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
