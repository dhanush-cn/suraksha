"""Smoke-verification of the background-worker layer. Run: python verify_workers.py

Exercises the arq task functions directly with a mocked ArqRedis, and the
enqueue helpers with a real (but faked-down) Redis connection. Verifies:

* idempotency SET NX behavior on duplicate alerts
* exponential-backoff retry via arq.Retry
* dead-letter LPUSH after the retry cap
* dispatch skips simulated-only paths cleanly when SMTP/Twilio are unset
* enqueue helper degrades to None when Redis is unreachable
* weather_service falls back gracefully on HTTP failure

Does NOT need a running arq worker or Redis instance -- the ctx dict is
built by hand with an AsyncMock 'redis'.
"""

from __future__ import annotations

import asyncio
import json
import os
import traceback
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("JWT_SECRET", "a" * 48)
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://u:p@localhost:5432/db")
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str):
    def wrapper(fn):
        try:
            if asyncio.iscoroutinefunction(fn):
                asyncio.run(fn())
            else:
                fn()
            results.append((PASS, name, ""))
        except Exception as exc:  # noqa: BLE001
            results.append((FAIL, name, f"{type(exc).__name__}: {exc}"))
            traceback.print_exc()
        return fn

    return wrapper


# ---------------------------------------------------------------------------
# dispatch_alert
# ---------------------------------------------------------------------------


def _sample_payload(**overrides: Any) -> dict[str, Any]:
    payload = {
        "idempotency_key": "test_key_abc123",
        "mine_id": 1,
        "mine_name": "Grasberg Open-Pit Mine",
        "risk_percentage": 88.0,
        "risk_level": "Critical",
        "top_shap_reason": "Pore Water Pressure (90.2 kPa)",
        "contact_email": None,  # -> simulated email path
        "contact_phone": None,  # -> simulated sms path
    }
    payload.update(overrides)
    return payload


@check("dispatch: first attempt acquires idempotency lock and delivers")
def _():
    from app.workers.tasks import dispatch_alert

    async def go():
        fake_redis = AsyncMock()
        fake_redis.set.return_value = True  # lock acquired
        result = await dispatch_alert({"redis": fake_redis, "job_try": 1}, _sample_payload())
        assert result["status"] == "delivered", result
        assert result["attempts"] == 1
        # SET NX call
        fake_redis.set.assert_awaited_once()
        args, kwargs = fake_redis.set.call_args
        assert kwargs.get("nx") is True
        assert kwargs.get("ex") == 3600

    asyncio.run(go())


@check("dispatch: duplicate on first attempt skips (SET NX returned False)")
def _():
    from app.workers.tasks import dispatch_alert

    async def go():
        fake_redis = AsyncMock()
        fake_redis.set.return_value = False  # lock already held by earlier job
        result = await dispatch_alert({"redis": fake_redis, "job_try": 1}, _sample_payload())
        assert result["status"] == "skipped", result
        assert result["reason"] == "duplicate"
        # Skipped path must not LPUSH to DLQ.
        fake_redis.lpush.assert_not_called()

    asyncio.run(go())


@check("dispatch: raises arq.Retry with exponential backoff on transient failure")
def _():
    from arq import Retry

    from app.workers.tasks import dispatch_alert

    async def go():
        fake_redis = AsyncMock()
        fake_redis.set.return_value = True

        with patch("app.workers.tasks._dispatch_email", side_effect=RuntimeError("SMTP boom")):
            # arq stores defer_score in *milliseconds* internally, so
            # Retry(defer=2) -> defer_score == 2000.
            # Attempt 1 -> Retry(defer=2 seconds)
            try:
                await dispatch_alert({"redis": fake_redis, "job_try": 1}, _sample_payload())
            except Retry as r:
                assert r.defer_score == 2_000, f"expected defer=2s (2000ms), got {r.defer_score}"
            else:
                raise AssertionError("expected arq.Retry on attempt 1")

            # Attempt 2 -> Retry(defer=4 seconds)
            try:
                await dispatch_alert({"redis": fake_redis, "job_try": 2}, _sample_payload())
            except Retry as r:
                assert r.defer_score == 4_000, f"expected defer=4s (4000ms), got {r.defer_score}"
            else:
                raise AssertionError("expected arq.Retry on attempt 2")

    asyncio.run(go())


@check("dispatch: dead-letters after DISPATCH_MAX_TRIES failures")
def _():
    from app.workers.tasks import DEAD_LETTER_KEY, DISPATCH_MAX_TRIES, dispatch_alert

    async def go():
        fake_redis = AsyncMock()
        fake_redis.set.return_value = True

        with patch("app.workers.tasks._dispatch_email", side_effect=RuntimeError("SMTP still boom")):
            result = await dispatch_alert(
                {"redis": fake_redis, "job_try": DISPATCH_MAX_TRIES},
                _sample_payload(),
            )
        assert result["status"] == "dead_lettered", result
        assert result["attempts"] == DISPATCH_MAX_TRIES
        # LPUSH into the DLQ, then LTRIM to bound the list.
        fake_redis.lpush.assert_awaited_once()
        lpush_args = fake_redis.lpush.call_args.args
        assert lpush_args[0] == DEAD_LETTER_KEY
        payload = json.loads(lpush_args[1])
        assert payload["error"].startswith("RuntimeError")
        assert payload["attempts"] == DISPATCH_MAX_TRIES
        fake_redis.ltrim.assert_awaited_once()

    asyncio.run(go())


@check("dispatch: simulates SMTP + Twilio cleanly when credentials absent")
def _():
    from app.workers.tasks import dispatch_alert

    async def go():
        fake_redis = AsyncMock()
        fake_redis.set.return_value = True
        # Ensure no live credentials in test env.
        for k in (
            "SMTP_HOST", "SMTP_SERVER", "SMTP_USER", "SMTP_PASSWORD", "SMTP_PASS",
            "TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM_NUMBER", "TWILIO_PHONE_NUMBER",
        ):
            os.environ.pop(k, None)
        result = await dispatch_alert({"redis": fake_redis, "job_try": 1}, _sample_payload())
        assert result["status"] == "delivered", result
        assert result["email"]["status"] == "simulated"
        assert result["sms"]["status"] == "simulated"

    asyncio.run(go())


# ---------------------------------------------------------------------------
# score_csv / analyze_image plumbing
# ---------------------------------------------------------------------------


@check("score_csv: reports error for missing file (no crash)")
def _():
    from app.workers.tasks import score_csv

    async def go():
        result = await score_csv({}, "/tmp/does-not-exist.csv", "phantom.csv")
        assert result["status"] == "error", result
        assert "no longer exists" in result["message"]

    asyncio.run(go())


@check("analyze_image: reports error for missing file (no crash)")
def _():
    from app.workers.tasks import analyze_image

    async def go():
        result = await analyze_image({}, "/tmp/does-not-exist.bin")
        assert result["status"] == "error", result
        assert "no longer exists" in result["message"]

    asyncio.run(go())


# ---------------------------------------------------------------------------
# queue helpers
# ---------------------------------------------------------------------------


@check("queue: enqueue returns None when Redis pool cannot be built")
def _():
    from app.workers import queue

    async def go():
        # Reset any cached pool and force get_pool to fail.
        queue._pool = None
        with patch("app.workers.queue.create_pool", side_effect=ConnectionRefusedError("no redis")):
            job_id = await queue.enqueue("dispatch_alert", {})
        assert job_id is None
        # Cached pool must remain unset so a subsequent call retries.
        assert queue._pool is None

    asyncio.run(go())


@check("queue: job_status returns 'unavailable' when queue is down")
def _():
    from app.workers import queue

    async def go():
        queue._pool = None
        with patch("app.workers.queue.create_pool", side_effect=ConnectionRefusedError("no redis")):
            status = await queue.job_status("some_id")
        assert status["status"] == "unavailable", status

    asyncio.run(go())


# ---------------------------------------------------------------------------
# weather_service async client
# ---------------------------------------------------------------------------


@check("weather: async client returns fallback payload on HTTP failure")
def _():
    from weather_service import fetch_open_meteo_weather

    async def go():
        # Coerce httpx to raise: mock AsyncClient.get to blow up.
        with patch("httpx.AsyncClient") as MockClient:
            instance = MockClient.return_value.__aenter__.return_value
            instance.get = AsyncMock(side_effect=RuntimeError("network down"))
            result = await fetch_open_meteo_weather(-4.05, 137.11)
        assert result["status"] == "fallback", result
        assert result["source"] == "Local Fallback Sensor"

    asyncio.run(go())


@check("weather: parses successful Open-Meteo response into flat schema")
def _():
    from weather_service import fetch_open_meteo_weather

    fake_response = MagicMock()
    fake_response.json.return_value = {
        "current": {
            "precipitation": 3.5,
            "relative_humidity_2m": 78.0,
            "temperature_2m": 21.5,
            "surface_pressure": 1009.0,
            "wind_speed_10m": 15.5,
        },
        "hourly": {"precipitation": [1.0, 1.2, 0.5, 0.3, 0.0, 0.0]},
    }
    fake_response.raise_for_status = MagicMock()

    async def go():
        with patch("httpx.AsyncClient") as MockClient:
            instance = MockClient.return_value.__aenter__.return_value
            instance.get = AsyncMock(return_value=fake_response)
            result = await fetch_open_meteo_weather(-4.05, 137.11)
        assert result["status"] == "success", result
        assert result["rainfall_mm"] == 3.5
        assert result["humidity_pct"] == 78.0
        # 6h antecedent sum from the hourly array
        assert abs(result["rain_rolling_6h"] - 3.0) < 1e-6

    asyncio.run(go())


# ---------------------------------------------------------------------------
# arq WorkerSettings loads correctly
# ---------------------------------------------------------------------------


@check("worker_settings: WorkerSettings exposes tasks and Redis settings")
def _():
    from arq.worker import Function

    from app.workers.settings import WorkerSettings
    from app.workers.tasks import DISPATCH_MAX_TRIES

    # WorkerSettings.functions is a mix of raw coroutines and arq.func(...)
    # wrappers -- unwrap both to a set of names for the assertion.
    def _name(entry):
        return entry.name if isinstance(entry, Function) else entry.__name__

    names = {_name(f) for f in WorkerSettings.functions}
    assert names == {"score_csv", "analyze_image", "dispatch_alert"}, names

    # Per-function overrides: max_tries=1 for the deterministic tasks.
    by_name = {_name(f): f for f in WorkerSettings.functions}
    assert by_name["score_csv"].max_tries == 1
    assert by_name["analyze_image"].max_tries == 1

    assert WorkerSettings.max_tries == DISPATCH_MAX_TRIES
    assert WorkerSettings.job_timeout == 300
    assert WorkerSettings.redis_settings.host  # non-empty


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
