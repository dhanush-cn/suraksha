"""Smoke-verification of the foundational layer. Run: python verify_core.py"""

from __future__ import annotations

import os
import traceback

os.environ.setdefault("JWT_SECRET", "a" * 48)
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://u:p@localhost:5432/db")
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("LOG_JSON", "true")
os.environ.setdefault("CORS_ORIGINS", "http://localhost:5173,http://localhost:8000")

from app.schemas.telemetry import SensorReading  # module scope: needed for route annotations

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


# ---------------------------------------------------------------- config ---
@check("config: loads and validates from env")
def _():
    from app.core.config import Environment, get_settings

    s = get_settings()
    assert s.environment is Environment.TEST
    assert s.jwt_secret.get_secret_value() == "a" * 48
    assert s.cors_origin_strings == ["http://localhost:5173", "http://localhost:8000"]
    assert s.api_prefix == "/api/v1"


@check("config: rejects short jwt_secret")
def _():
    from pydantic import ValidationError

    from app.core.config import Settings

    try:
        Settings(jwt_secret="tooshort", database_url="postgresql://u:p@h/d")  # type: ignore[call-arg]
    except ValidationError as exc:
        assert "at least 32 characters" in str(exc)
    else:
        raise AssertionError("short secret was accepted")


@check("config: production rejects debug=True")
def _():
    from pydantic import ValidationError

    from app.core.config import Settings

    try:
        Settings(  # type: ignore[call-arg]
            jwt_secret="b" * 48,
            database_url="postgresql://u:p@h/d",
            environment="production",
            debug=True,
            cors_origins=["https://app.example.com"],
            trusted_hosts=["app.example.com"],
        )
    except ValidationError as exc:
        assert "debug must be False in production" in str(exc)
    else:
        raise AssertionError("production debug=True was accepted")


@check("config: production rejects empty cors_origins")
def _():
    from pydantic import ValidationError

    from app.core.config import Settings

    saved = os.environ.pop("CORS_ORIGINS", None)
    try:
        Settings(  # type: ignore[call-arg]
            jwt_secret="b" * 48,
            database_url="postgresql://u:p@h/d",
            environment="production",
            trusted_hosts=["app.example.com"],
        )
    except ValidationError as exc:
        assert "cors_origins must be set explicitly" in str(exc)
    else:
        raise AssertionError("production empty CORS was accepted")
    finally:
        if saved is not None:
            os.environ["CORS_ORIGINS"] = saved


# -------------------------------------------------------------- security ---
@check("security: bcrypt hash/verify round-trip")
def _():
    from app.core.security import hash_password, verify_password

    hashed = hash_password("correct-horse-battery", rounds=10)
    assert verify_password("correct-horse-battery", hashed)
    assert not verify_password("wrong-password", hashed)
    assert not verify_password("anything", None)  # unknown-user path


@check("security: JWT round-trip with pinned algorithm")
def _():
    from app.core.config import get_settings
    from app.core.security import Role, TokenType, create_token, decode_token

    s = get_settings()
    token, jti, _exp = create_token(settings=s, subject="42", role=Role.OPERATOR, mine_id=1)
    claims = decode_token(token, settings=s)
    assert claims["sub"] == "42"
    assert claims["role"] == "operator"
    assert claims["mine_id"] == 1
    assert claims["jti"] == jti
    assert claims["iss"] == s.jwt_issuer


@check("security: alg=none forgery is rejected")
def _():
    import jwt as pyjwt

    from app.core.config import get_settings
    from app.core.exceptions import TokenInvalidError
    from app.core.security import decode_token

    s = get_settings()
    forged = pyjwt.encode(
        {"sub": "1", "role": "admin", "jti": "x", "iss": s.jwt_issuer, "aud": s.jwt_audience},
        key="",
        algorithm="none",
    )
    try:
        decode_token(forged, settings=s)
    except TokenInvalidError:
        pass
    else:
        raise AssertionError("alg=none token was accepted!")


@check("security: refresh token rejected where access expected")
def _():
    from app.core.config import get_settings
    from app.core.exceptions import TokenInvalidError
    from app.core.security import Role, TokenType, create_token, decode_token

    s = get_settings()
    token, _, _ = create_token(
        settings=s, subject="1", role=Role.ADMIN, token_type=TokenType.REFRESH
    )
    try:
        decode_token(token, settings=s, expected_type=TokenType.ACCESS)
    except TokenInvalidError:
        pass
    else:
        raise AssertionError("refresh token accepted as access token!")


@check("security: role hierarchy")
def _():
    from app.core.security import Role

    assert Role.ADMIN.satisfies(Role.OPERATOR)
    assert not Role.VIEWER.satisfies(Role.ADMIN)


# --------------------------------------------------------------- schemas ---
@check("schemas: sensor bounds reject impossible latitude")
def _():
    from pydantic import ValidationError

    from app.schemas.mine import MineCreate

    try:
        MineCreate(
            name="X", company="Y", location_name="Z",
            latitude=9999.0, longitude=0.0, contact_email="a@b.com",
        )
    except ValidationError:
        pass
    else:
        raise AssertionError("latitude 9999 accepted")


@check("schemas: NaN rejected in sensor reading")
def _():
    from pydantic import ValidationError

    from app.schemas.telemetry import SensorReading

    try:
        SensorReading(mine_id=1, velocity_mm_h=float("nan"))
    except ValidationError:
        pass
    else:
        raise AssertionError("NaN velocity accepted — would yield risk=nan")


@check("schemas: extra fields forbidden on requests")
def _():
    from pydantic import ValidationError

    from app.schemas.telemetry import SensorReading

    try:
        SensorReading(mine_id=1, is_admin=True)  # type: ignore[call-arg]
    except ValidationError:
        pass
    else:
        raise AssertionError("unknown field silently accepted (mass assignment)")


@check("schemas: mine requires a contact channel")
def _():
    from pydantic import ValidationError

    from app.schemas.mine import MineCreate

    try:
        MineCreate(name="X", company="Y", location_name="Z", latitude=0.0, longitude=0.0)
    except ValidationError as exc:
        assert "contact_email or contact_phone" in str(exc)
    else:
        raise AssertionError("mine with no alert destination accepted")


@check("schemas: rain_rolling_6h derived and coherence-checked")
def _():
    from pydantic import ValidationError

    from app.schemas.telemetry import SensorReading

    reading = SensorReading(mine_id=1, rainfall_mm=10.0)
    assert reading.rain_rolling_6h == 25.0
    assert reading.recorded_at is not None
    try:
        SensorReading(mine_id=1, rainfall_mm=10.0, rain_rolling_6h=2.0)
    except ValidationError:
        pass
    else:
        raise AssertionError("incoherent 6h rainfall accepted")


@check("schemas: out-of-distribution flag")
def _():
    from app.schemas.telemetry import SensorReading

    assert SensorReading(mine_id=1, velocity_mm_h=500.0).is_out_of_distribution
    assert not SensorReading(mine_id=1, velocity_mm_h=0.05).is_out_of_distribution


@check("schemas: Principal blocks cross-tenant access")
def _():
    from app.core.exceptions import TenantAccessDeniedError
    from app.core.security import Role
    from app.schemas.auth import Principal
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    operator = Principal(
        user_id="7", username="grasberg_user", role=Role.OPERATOR, mine_id=1,
        token_id="j", issued_at=now, expires_at=now,
    )
    operator.authorize_mine(1)  # own mine: allowed
    try:
        operator.authorize_mine(2)
    except TenantAccessDeniedError:
        pass
    else:
        raise AssertionError("cross-tenant access allowed!")

    admin = Principal(
        user_id="1", username="admin", role=Role.ADMIN, mine_id=None,
        token_id="j", issued_at=now, expires_at=now,
    )
    admin.authorize_mine(2)  # admin unscoped: allowed


@check("schemas: non-admin without mine_id scope is rejected")
def _():
    from datetime import datetime, timezone

    from pydantic import ValidationError

    from app.core.security import Role
    from app.schemas.auth import Principal

    now = datetime.now(timezone.utc)
    try:
        Principal(
            user_id="7", username="u", role=Role.OPERATOR, mine_id=None,
            token_id="j", issued_at=now, expires_at=now,
        )
    except ValidationError:
        pass
    else:
        raise AssertionError("unscoped operator accepted — would bypass isolation")


@check("schemas: alert idempotency key is stable within a minute bucket")
def _():
    from datetime import datetime, timezone

    from app.schemas.alert import AlertCreate
    from app.schemas.telemetry import RiskLevel

    base = dict(
        mine_id=1, risk_percentage=88.0, risk_level=RiskLevel.CRITICAL,
        rainfall_mm=40.0, pore_pressure_kpa=95.0, velocity_mm_h=5.0,
        seismic_rms_g=0.4, top_shap_reason="pore pressure",
    )
    a = AlertCreate(**base, triggered_at=datetime(2026, 8, 19, 10, 30, 5, tzinfo=timezone.utc))
    b = AlertCreate(**base, triggered_at=datetime(2026, 8, 19, 10, 30, 55, tzinfo=timezone.utc))
    c = AlertCreate(**base, triggered_at=datetime(2026, 8, 19, 10, 31, 5, tzinfo=timezone.utc))
    assert a.idempotency_key == b.idempotency_key, "same minute must dedupe"
    assert a.idempotency_key != c.idempotency_key, "next minute must be a new alert"


@check("schemas: dispatch state machine consistency")
def _():
    from datetime import datetime, timezone

    from pydantic import ValidationError

    from app.schemas.alert import AlertDispatch, DispatchChannel, DispatchStatus

    now = datetime.now(timezone.utc)
    try:
        AlertDispatch(
            id=1, alert_id=1, channel=DispatchChannel.SMS,
            recipient_masked="+1***99", status=DispatchStatus.SENT,
            created_at=now, delivered_at=None,
        )
    except ValidationError:
        pass
    else:
        raise AssertionError("SENT without delivered_at accepted")


# ------------------------------------------------------- app integration ---
@check("app: middleware + error handlers produce uniform envelope")
def _():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.core.config import get_settings
    from app.core.error_handlers import install_exception_handlers
    from app.core.exceptions import NotFoundError, TenantAccessDeniedError
    from app.core.logging import configure_logging
    from app.core.middleware import CORRELATION_HEADER, install_middleware
    settings = get_settings()
    configure_logging(settings)
    app = FastAPI()
    install_middleware(app, settings)
    install_exception_handlers(app, settings)

    @app.get("/boom")
    def boom():
        raise NotFoundError("Mine", identifier=42)

    @app.get("/denied")
    def denied():
        raise TenantAccessDeniedError(mine_id=2)

    @app.get("/crash")
    def crash():
        return 1 / 0

    @app.post("/reading")
    def reading(payload: SensorReading):
        return {"ok": True, "risk_input": payload.velocity_mm_h}

    client = TestClient(app, raise_server_exceptions=False)

    r = client.get("/boom")
    assert r.status_code == 404, r.status_code
    body = r.json()
    assert body["error"]["code"] == "not_found"
    assert body["error"]["correlation_id"], "correlation id missing from envelope"
    assert r.headers[CORRELATION_HEADER]
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["X-Frame-Options"] == "DENY"

    r = client.get("/denied")
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "tenant_access_denied"

    r = client.get("/crash")
    assert r.status_code == 500
    # ZeroDivisionError detail is allowed in non-production only.
    assert r.json()["error"]["code"] == "internal_error"
    assert "stacktrace" not in r.text.lower()

    r = client.post("/reading", json={"mine_id": 1, "velocity_mm_h": 0.5})
    assert r.status_code == 200, r.text

    r = client.post("/reading", json={"mine_id": 1, "velocity_mm_h": 99999.0})
    assert r.status_code == 422
    fields = r.json()["error"]["details"]["fields"]
    assert any(f["field"] == "velocity_mm_h" for f in fields), fields

    r = client.post("/reading", json={"mine_id": 0})
    assert r.status_code == 422


@check("app: body size limit returns 413")
def _():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.core.config import get_settings
    from app.core.error_handlers import install_exception_handlers
    from app.core.middleware import BodySizeLimitMiddleware

    app = FastAPI()
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=100)
    install_exception_handlers(app, get_settings())

    @app.post("/upload")
    def upload(payload: dict):
        return {"ok": True}

    client = TestClient(app, raise_server_exceptions=False)
    r = client.post("/upload", json={"data": "x" * 5_000})
    assert r.status_code == 413, r.status_code
    assert r.json()["error"]["code"] == "payload_too_large"


@check("logging: JSON output redacts secrets")
def _():
    import json
    import logging
    from io import StringIO

    from app.core.config import get_settings
    from app.core.logging import JsonFormatter, set_correlation_id

    stream = StringIO()
    handler = logging.StreamHandler(stream)
    s = get_settings()
    handler.setFormatter(
        JsonFormatter(service=s.project_name, environment="test", version=s.version)
    )
    logger = logging.getLogger("verify.redaction")
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False

    set_correlation_id("abc123")
    logger.info("login attempt", extra={"username": "bob", "password": "hunter2"})

    record = json.loads(stream.getvalue().strip())
    assert record["password"] == "***redacted***", record
    assert record["username"] == "bob"
    assert record["correlation_id"] == "abc123"
    assert record["level"] == "INFO"


if __name__ == "__main__":
    print("\n" + "=" * 78)
    for status, name, detail in results:
        marker = "\u2713" if status == PASS else "\u2717"
        print(f"  {marker} {status}  {name}")
        if detail:
            print(f"           -> {detail}")
    failures = sum(1 for status, _, _ in results if status == FAIL)
    print("=" * 78)
    print(f"  {len(results) - failures}/{len(results)} checks passed")
    print("=" * 78)
    raise SystemExit(1 if failures else 0)