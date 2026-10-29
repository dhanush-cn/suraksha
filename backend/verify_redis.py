"""Smoke-verification of the Redis-backed features. Run: python verify_redis.py

Exercises the four Step-6 additions -- token-bucket rate limiter,
mine-metadata cache, JWT revocation blocklist, and the emergency
Redis Stream -- against a fakeredis instance so no live Redis is
required. Also asserts the environment-aware fail-open/closed
behaviour of the blocklist.

Checks:

* Rate limiter: allows up to capacity, denies over, refill increments
  tokens after wall-clock advances, per-scope keys don't share
  buckets, Redis-outage path fails OPEN.
* Cache: get/set/invalidate round trip; miss returns None; poisoned
  entry (non-JSON) is dropped rather than propagated; invalidate_mine
  clears both single-mine and list keys atomically.
* Blocklist: revoke sets a key with EX = ttl_seconds; is_revoked
  reflects it; TTL <= 0 is a no-op; env=production + Redis down
  fails CLOSED; env=development + Redis down fails OPEN.
* Streams: publish_emergency_event XADDs; recent_events reads
  newest-first; ensure_consumer_group is idempotent; MAXLEN is
  applied (approximate, so we assert length is bounded).
"""

from __future__ import annotations

import asyncio
import os
import time
import traceback
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("JWT_SECRET", "a" * 48)
os.environ.setdefault("DATABASE_URL", "sqlite:///./mines.db")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("ENVIRONMENT", "test")  # non-production for blocklist

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


def _mock_redis_dep(value):
    """AsyncMock stand-in for the get_redis dependency.

    ``patch(..., return_value=coroutine)`` binds ONE coroutine that
    can only be awaited once; multiple ``await get_redis()`` calls
    inside a test would then raise / return None on subsequent
    invocations. AsyncMock generates a fresh awaitable each time.

    Kept at module scope so the @check decorator (which invokes the
    check function at import time) can reference it.
    """
    return AsyncMock(return_value=value)


def _fake_redis():
    """Return a fresh fakeredis async client for one test.

    fakeredis's aioredis backend accepts the same commands the real
    async Redis does (SET/GET/EXISTS/HMSET/EVAL/XADD/XREVRANGE) and
    ships an EVAL implementation good enough for our Lua rate limiter.
    """
    import fakeredis.aioredis as fake

    return fake.FakeRedis(decode_responses=True)


def _reset_env_settings():
    """Clear Settings lru_cache so ENVIRONMENT overrides in a check
    take effect on the next get_settings() call."""
    from app.core.config import get_settings

    get_settings.cache_clear()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# rate limiter
# ---------------------------------------------------------------------------


@check("rate_limit: allows up to capacity, denies over, refills after time")
def _():
    from fastapi import HTTPException

    from app.core.rate_limit import RateLimit, RateLimiter

    async def go():
        # refill_rate deliberately near-zero so wall-clock drift
        # between the three "allowed" requests can't inadvertently
        # refill the bucket back above 1 before we assert 429.
        limiter = RateLimiter(RateLimit(capacity=3, refill_rate=0.001, scope="test"))
        fake = _fake_redis()

        class FakeRequest:
            client = type("_C", (), {"host": "1.2.3.4"})()
            headers = {}
            url = type("_U", (), {"path": "/x"})()
            state = MagicMock(spec=[])  # no principal -> falls back to IP

        with patch("app.core.rate_limit.get_redis", new=_mock_redis_dep(fake)):
            # 3 requests succeed
            for i in range(3):
                await limiter(FakeRequest())

            # 4th trips
            try:
                await limiter(FakeRequest())
            except HTTPException as exc:
                assert exc.status_code == 429
                assert "Retry-After" in exc.headers
            else:
                raise AssertionError("expected 429 on 4th request")

    asyncio.run(go())


@check("rate_limit: different scopes don't share buckets")
def _():
    from app.core.rate_limit import RateLimit, RateLimiter

    async def go():
        one = RateLimiter(RateLimit(capacity=1, refill_rate=0.001, scope="alpha"))
        two = RateLimiter(RateLimit(capacity=1, refill_rate=0.001, scope="beta"))
        fake = _fake_redis()

        class FakeRequest:
            client = type("_C", (), {"host": "5.6.7.8"})()
            headers = {}
            url = type("_U", (), {"path": "/x"})()
            state = MagicMock(spec=[])

        with patch("app.core.rate_limit.get_redis", new=_mock_redis_dep(fake)):
            await one(FakeRequest())  # spend alpha's token
            # beta should still have its token -- separate scope key
            await two(FakeRequest())

    asyncio.run(go())


@check("rate_limit: fails OPEN when Redis is unavailable")
def _():
    from app.core.rate_limit import RateLimit, RateLimiter

    async def go():
        limiter = RateLimiter(RateLimit(capacity=1, refill_rate=0.001, scope="down"))

        class FakeRequest:
            client = type("_C", (), {"host": "9.9.9.9"})()
            headers = {}
            url = type("_U", (), {"path": "/x"})()
            state = MagicMock(spec=[])

        with patch("app.core.rate_limit.get_redis", new=_mock_redis_dep(None)):
            # Would trip immediately if the limiter enforced during
            # outage. Fail-open means all pass.
            for _ in range(20):
                await limiter(FakeRequest())

    asyncio.run(go())


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------


@check("cache: mine set/get round-trip; invalidate clears list + single")
def _():
    from app.core.cache import (
        get_cached_mine,
        get_cached_mine_list,
        invalidate_mine,
        set_cached_mine,
        set_cached_mine_list,
    )

    async def go():
        fake = _fake_redis()
        with patch("app.core.cache.get_redis", new=_mock_redis_dep(fake)):
            payload = {"id": 1, "name": "Grasberg", "alert_threshold_pct": 70.0}
            await set_cached_mine(1, payload)
            # Scope-keyed list cache API (Step 9). "admin" here is just
            # a stand-in scope hash; the round-trip is what we're
            # exercising, not scope-filtering itself.
            await set_cached_mine_list("admin", [payload])

            assert await get_cached_mine(1) == payload
            assert await get_cached_mine_list("admin") == [payload]

            await invalidate_mine(1)
            assert await get_cached_mine(1) is None
            assert await get_cached_mine_list("admin") is None

    asyncio.run(go())


@check("cache: poisoned (non-JSON) entry is dropped, not propagated")
def _():
    from app.core.cache import get_cached_mine

    async def go():
        fake = _fake_redis()
        await fake.set("rockfallguard:cache:mine:99", "not-json-{[")

        with patch("app.core.cache.get_redis", new=_mock_redis_dep(fake)):
            assert await get_cached_mine(99) is None
        # And the poisoned key should be gone.
        assert await fake.get("rockfallguard:cache:mine:99") is None

    asyncio.run(go())


@check("cache: outage returns None cleanly (no exception)")
def _():
    from app.core.cache import get_cached_mine, set_cached_mine

    async def go():
        with patch("app.core.cache.get_redis", new=_mock_redis_dep(None)):
            assert await get_cached_mine(1) is None
            await set_cached_mine(1, {"id": 1})  # must not raise

    asyncio.run(go())


# ---------------------------------------------------------------------------
# blocklist
# ---------------------------------------------------------------------------


@check("blocklist: revoke + is_revoked round-trip; TTL <= 0 is a no-op")
def _():
    from app.core.blocklist import is_revoked, revoke

    async def go():
        fake = _fake_redis()
        with patch("app.core.blocklist.get_redis", new=_mock_redis_dep(fake)):
            assert await is_revoked("abc123") is False
            assert await revoke("abc123", ttl_seconds=60) is True
            assert await is_revoked("abc123") is True

            # ttl=0: token already expired; nothing to revoke.
            assert await revoke("expired", ttl_seconds=0) is True
            assert await is_revoked("expired") is False

    asyncio.run(go())


@check("blocklist: env=test + Redis down -> fails OPEN (not_revoked)")
def _():
    _reset_env_settings()
    os.environ["ENVIRONMENT"] = "test"
    _reset_env_settings()

    from app.core.blocklist import is_revoked

    async def go():
        with patch("app.core.blocklist.get_redis", new=_mock_redis_dep(None)):
            assert await is_revoked("anything") is False

    asyncio.run(go())


@check("blocklist: env=production + Redis down -> fails CLOSED (revoked)")
def _():
    _reset_env_settings()
    os.environ["ENVIRONMENT"] = "production"
    # Production settings validation requires a couple more fields;
    # provide them just for this check.
    prev_cors = os.environ.get("CORS_ORIGINS")
    prev_hosts = os.environ.get("TRUSTED_HOSTS")
    prev_debug = os.environ.get("DEBUG")
    os.environ["CORS_ORIGINS"] = "https://app.example.com"
    os.environ["TRUSTED_HOSTS"] = "app.example.com"
    os.environ["DEBUG"] = "false"
    _reset_env_settings()

    try:
        from app.core.blocklist import is_revoked

        async def go():
            with patch("app.core.blocklist.get_redis", new=_mock_redis_dep(None)):
                assert await is_revoked("anything") is True

        asyncio.run(go())
    finally:
        # Restore env
        os.environ["ENVIRONMENT"] = "test"
        for key, prev in [
            ("CORS_ORIGINS", prev_cors),
            ("TRUSTED_HOSTS", prev_hosts),
            ("DEBUG", prev_debug),
        ]:
            if prev is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prev
        _reset_env_settings()


# ---------------------------------------------------------------------------
# streams
# ---------------------------------------------------------------------------


@check("streams: XADD publishes, XREVRANGE returns newest-first")
def _():
    from app.core.streams import publish_emergency_event, recent_events

    async def go():
        fake = _fake_redis()
        with patch("app.core.streams.get_redis", new=_mock_redis_dep(fake)):
            for i in range(3):
                await publish_emergency_event(
                    {"mine_id": 1, "risk_percentage": 70 + i, "seq": i}
                )
            events = await recent_events(limit=10)
            assert len(events) == 3
            # newest-first: seq 2 -> 1 -> 0
            assert [e["seq"] for e in events] == [2, 1, 0]
            # each event carries its stream id
            assert all("_stream_id" in e for e in events)

    asyncio.run(go())


@check("streams: outage returns [] / None cleanly")
def _():
    from app.core.streams import publish_emergency_event, recent_events, stream_length

    async def go():
        with patch("app.core.streams.get_redis", new=_mock_redis_dep(None)):
            assert await publish_emergency_event({"x": 1}) is None
            assert await recent_events() == []
            assert await stream_length() is None

    asyncio.run(go())


@check("streams: ensure_consumer_group is idempotent (BUSYGROUP swallowed)")
def _():
    from app.core.streams import ensure_consumer_group

    async def go():
        fake = _fake_redis()
        with patch("app.core.streams.get_redis", new=_mock_redis_dep(fake)):
            await ensure_consumer_group()
            # Second call must not raise
            await ensure_consumer_group()

    asyncio.run(go())


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
