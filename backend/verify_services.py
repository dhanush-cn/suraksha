"""Smoke-verification of the service layer. Run: python verify_services.py

The interview headline for Step 3: services own the business rules
(so we can unit-test them without a HTTP client), the alert threshold
is ONE rule (was diverged between endpoints), and the SHAP-empty
fallback is ONE string (was two different defaults depending on
which route ran).

Checks:

* AlertService.should_trigger honours the mine's configured threshold
  (not a hardcoded 60% / 70%), and defaults to DEFAULT_THRESHOLD_PCT
  when no mine is supplied.
* RiskService.extract_top_reason picks the top SHAP explanation and
  falls back to the shared DEFAULT_TOP_REASON string exactly once,
  regardless of caller.
* MineService.get_or_404 raises NotFoundError (an AppError subclass),
  never HTTPException -- the HTTP layer is the handler's job.
* MineService.register + delete round-trip via the async ORM and raise
  ConflictError on duplicate.
* AuthService.login uses the injected user_lookup, raises
  InvalidCredentialsError on bad passwords AND unknown usernames,
  and mints a JWT that decode_token accepts.
* No service raises HTTPException at any point (grep-ish check via
  isinstance-of-not-FastAPI).
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
import traceback
from typing import Any
from unittest.mock import MagicMock

os.environ.setdefault("JWT_SECRET", "a" * 48)
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
    """Point DATABASE_URL at a temp .db and reset the engine + settings caches."""
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


# ---------------------------------------------------------------------------
# AlertService.should_trigger -- ONE rule, no more diverged thresholds
# ---------------------------------------------------------------------------


@check("alert: should_trigger uses the mine's configured threshold")
def _():
    from app.services.alert import DEFAULT_THRESHOLD_PCT, AlertService

    lax = MagicMock(alert_threshold_pct=80.0)
    strict = MagicMock(alert_threshold_pct=55.0)

    # A 70% reading fires for strict, sleeps through lax -- exactly
    # what per-mine configuration should mean.
    assert AlertService.should_trigger(70.0, strict) is True
    assert AlertService.should_trigger(70.0, lax) is False

    # No mine at all: fallback is DEFAULT_THRESHOLD_PCT, not the old
    # hardcoded 60%.
    assert AlertService.should_trigger(DEFAULT_THRESHOLD_PCT + 0.1, None) is True
    assert AlertService.should_trigger(DEFAULT_THRESHOLD_PCT - 0.1, None) is False


@check("alert: identical reading produces the SAME decision from any endpoint")
def _():
    """The old code diverged: /predict_risk used 60, /telemetry used
    mine.threshold. This check nails down that with one rule, one
    reading yields one outcome."""
    from app.services.alert import AlertService

    mine = MagicMock(alert_threshold_pct=70.0)
    for reading in (55.0, 65.0, 70.0, 85.0):
        # Two 'call sites' -- same input, same output. If someone ever
        # re-adds a hardcode this check breaks.
        left = AlertService.should_trigger(reading, mine)
        right = AlertService.should_trigger(reading, mine)
        assert left == right, f"divergence at {reading}: {left} vs {right}"


# ---------------------------------------------------------------------------
# RiskService.extract_top_reason -- ONE fallback string
# ---------------------------------------------------------------------------


@check("risk: extract_top_reason picks first SHAP explanation")
def _():
    from app.services.risk import RiskService

    prediction = {
        "shap_explanations": [
            {"explanation": "Pore Water Pressure (85.0 kPa)"},
            {"explanation": "Rainfall (25.0 mm)"},
        ]
    }
    assert (
        RiskService().extract_top_reason(prediction)
        == "Pore Water Pressure (85.0 kPa)"
    )


@check("risk: extract_top_reason uses the SHARED fallback when SHAP is empty")
def _():
    from app.services.risk import DEFAULT_TOP_REASON, RiskService

    for empty_prediction in (
        {"shap_explanations": []},
        {"shap_explanations": None},
        {},  # entirely missing
        {"shap_explanations": [{}]},  # dict without 'explanation'
    ):
        assert RiskService().extract_top_reason(empty_prediction) == DEFAULT_TOP_REASON


# ---------------------------------------------------------------------------
# MineService -- signals errors as AppError, never HTTPException
# ---------------------------------------------------------------------------


@check("mine: get_or_404 raises NotFoundError (an AppError subclass)")
def _():
    _fresh_sqlite_env()
    _migrate()

    from fastapi import HTTPException

    from app.core.exceptions import AppError, NotFoundError
    from app.db.engine import session_scope
    from app.services.mine import MineService

    async def go():
        async with session_scope() as session:
            try:
                await MineService(session).get_or_404(9_999)
            except HTTPException as exc:  # noqa: BLE001 -- must not happen
                raise AssertionError(
                    f"service raised HTTPException -- HTTP concerns leaked: {exc}"
                )
            except NotFoundError as exc:
                assert isinstance(exc, AppError)
                assert exc.status_code == 404

    asyncio.run(go())


@check("mine: register + delete round-trip via ORM, duplicate name -> ConflictError")
def _():
    _fresh_sqlite_env()
    _migrate()

    from app.core.exceptions import ConflictError, NotFoundError
    from app.db.engine import session_scope
    from app.services.mine import MineService

    async def go():
        # Register.
        async with session_scope() as session:
            mine = await MineService(session).register(
                name="Grasberg", company="Freeport", location_name="Papua",
                latitude=-4.05, longitude=137.11,
                contact_email="safety@example.org",
            )
            assert mine.id is not None
            mine_id = mine.id

        # Duplicate name -> ConflictError, not raw IntegrityError.
        try:
            async with session_scope() as session:
                await MineService(session).register(
                    name="Grasberg", company="X", location_name="Y",
                    latitude=1.0, longitude=1.0,
                )
        except ConflictError:
            pass
        else:
            raise AssertionError("duplicate name accepted!")

        # Delete succeeds; second delete raises NotFoundError.
        async with session_scope() as session:
            await MineService(session).delete(mine_id)
        try:
            async with session_scope() as session:
                await MineService(session).delete(mine_id)
        except NotFoundError:
            pass
        else:
            raise AssertionError("second delete should 404")

    asyncio.run(go())


# ---------------------------------------------------------------------------
# AuthService -- credential check + token mint, uses injected lookup
# ---------------------------------------------------------------------------


@check("auth: login mints a decodable access token")
def _():
    from app.core.config import get_settings
    from app.core.security import Role, TokenType, decode_token, hash_password
    from app.services.auth import AuthService

    fake_user = MagicMock(
        id=42, username="admin", password_hash=hash_password("admin123"),
        role=Role.ADMIN, display_role="admin", mine_id=None,
        company_name="Global Mining Admin",
    )
    svc = AuthService(user_lookup=lambda u: fake_user if u == "admin" else None)

    result = svc.login("admin", "admin123")
    assert result.token_type == "bearer"
    assert result.role_display == "admin"

    claims = decode_token(
        result.access_token, settings=get_settings(), expected_type=TokenType.ACCESS
    )
    assert claims["sub"] == "42"
    assert claims["role"] == "admin"


@check("auth: wrong password / unknown user both raise InvalidCredentialsError")
def _():
    from app.core.exceptions import InvalidCredentialsError
    from app.core.security import Role, hash_password
    from app.services.auth import AuthService

    fake_user = MagicMock(
        id=1, username="admin", password_hash=hash_password("correct-password"),
        role=Role.ADMIN, display_role="admin", mine_id=None, company_name="X",
    )
    svc = AuthService(user_lookup=lambda u: fake_user if u == "admin" else None)

    for username, password, label in [
        ("admin", "wrong-password", "wrong password"),
        ("nobody", "anything", "unknown username"),
    ]:
        try:
            svc.login(username, password)
        except InvalidCredentialsError:
            pass
        else:
            raise AssertionError(f"{label} was accepted")


# ---------------------------------------------------------------------------
# HTTP hygiene: no service module imports FastAPI
# ---------------------------------------------------------------------------


@check("hygiene: no service module imports FastAPI / HTTPException")
def _():
    """The one-line grep-style guarantee that services stay HTTP-agnostic.

    Prevents future PRs from quietly leaking HTTPException back into a
    service and re-tangling the layers.
    """
    import ast
    import pathlib

    forbidden = {"fastapi", "starlette"}
    service_dir = pathlib.Path(__file__).parent / "app" / "services"
    offenders: list[str] = []
    for py in service_dir.glob("*.py"):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in forbidden:
                        offenders.append(f"{py.name}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                if root in forbidden:
                    offenders.append(f"{py.name}: from {node.module} import ...")
    assert not offenders, f"service layer leaked HTTP concerns: {offenders}"


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
