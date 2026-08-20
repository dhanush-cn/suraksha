"""Redis-backed read cache with explicit invalidation.

Wraps expensive-to-produce, rarely-changing data (mine metadata, ML
feature-name manifests, etc.) so we don't pay the DB round trip on
every read.

Design choices:

* **Explicit invalidation, not TTL alone.** A stale mine record
  showing a wrong ``alert_threshold_pct`` for even 60s can produce a
  wrong alert decision. Every writer (register / update / delete)
  invalidates the affected keys. TTL is a safety net only.
* **Fail open on cache miss AND on Redis outage.** If Redis is down,
  the caller falls through to the source of truth (the DB) -- a slow
  API is preferable to a broken one. Every cache method logs the
  degradation so ops see it.
* **JSON encoding, not pickle.** Cache values cross process boundaries
  (API pods, worker pods, ops shells running ``redis-cli``) so the
  format must be human-inspectable and language-neutral. pickle would
  also be a supply-chain footgun (arbitrary code execution on unpickle
  from a compromised Redis).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Awaitable, Callable, TypeVar

from app.core.config import get_settings
from app.core.metrics import record_cache_hit, record_cache_miss
from app.core.redis_client import get_redis

logger = logging.getLogger(__name__)

T = TypeVar("T")

_MINE_KEY_PREFIX = "rockfallguard:cache:mine"
_MINE_LIST_KEY = f"{_MINE_KEY_PREFIX}:list"


def _mine_key(mine_id: int) -> str:
    return f"{_MINE_KEY_PREFIX}:{mine_id}"


async def get_cached_mine(mine_id: int) -> dict[str, Any] | None:
    """Read a single mine record from cache; ``None`` on miss / outage.

    Records to Prometheus: outage counts as a miss (from the caller's
    POV they still had to hit the DB), so hit-ratio metrics stay honest
    during a Redis outage instead of silently dropping the sample.
    """
    client = await get_redis()
    if client is None:
        record_cache_miss("mine")
        return None
    try:
        raw = await client.get(_mine_key(mine_id))
    except Exception as exc:  # noqa: BLE001
        logger.warning("cache read failed for mine %s: %s", mine_id, exc)
        record_cache_miss("mine")
        return None
    if raw is None:
        record_cache_miss("mine")
        return None
    try:
        payload = json.loads(raw)
        record_cache_hit("mine")
        return payload
    except (ValueError, TypeError):
        # Poisoned entry -- drop it so the next read repopulates.
        try:
            await client.delete(_mine_key(mine_id))
        except Exception:  # noqa: BLE001,S110 -- best effort
            pass
        record_cache_miss("mine")
        return None


async def get_cached_mine_list() -> list[dict[str, Any]] | None:
    client = await get_redis()
    if client is None:
        record_cache_miss("mine_list")
        return None
    try:
        raw = await client.get(_MINE_LIST_KEY)
    except Exception as exc:  # noqa: BLE001
        logger.warning("cache read failed for mine list: %s", exc)
        record_cache_miss("mine_list")
        return None
    if raw is None:
        record_cache_miss("mine_list")
        return None
    try:
        payload = json.loads(raw)
        record_cache_hit("mine_list")
        return payload
    except (ValueError, TypeError):
        try:
            await client.delete(_MINE_LIST_KEY)
        except Exception:  # noqa: BLE001,S110
            pass
        record_cache_miss("mine_list")
        return None


async def set_cached_mine(mine_id: int, payload: dict[str, Any]) -> None:
    client = await get_redis()
    if client is None:
        return
    try:
        await client.set(
            _mine_key(mine_id),
            json.dumps(payload, default=str),
            ex=get_settings().mine_cache_ttl_seconds,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("cache write failed for mine %s: %s", mine_id, exc)


async def set_cached_mine_list(payload: list[dict[str, Any]]) -> None:
    client = await get_redis()
    if client is None:
        return
    try:
        await client.set(
            _MINE_LIST_KEY,
            json.dumps(payload, default=str),
            ex=get_settings().mine_cache_ttl_seconds,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("cache write failed for mine list: %s", exc)


async def invalidate_mine(mine_id: int) -> None:
    """Drop a single mine and the list -- both are stale after a write."""
    client = await get_redis()
    if client is None:
        return
    try:
        # Pipeline both DELETEs into one round trip; the list is
        # invalidated together with any single-mine key because
        # `register` / `delete` / `update` all mutate the list too.
        pipe = client.pipeline()
        pipe.delete(_mine_key(mine_id))
        pipe.delete(_MINE_LIST_KEY)
        await pipe.execute()
    except Exception as exc:  # noqa: BLE001
        logger.warning("cache invalidate failed for mine %s: %s", mine_id, exc)


async def invalidate_all_mines() -> None:
    """Nuke the whole mine cache -- for maintenance ops (bulk import, etc.)."""
    client = await get_redis()
    if client is None:
        return
    try:
        # SCAN + DEL rather than KEYS + DEL: KEYS blocks the Redis
        # event loop for the duration of the scan, which on a big keyspace
        # is a foot-gun. SCAN is O(1) per iteration.
        cursor = 0
        while True:
            cursor, batch = await client.scan(
                cursor=cursor, match=f"{_MINE_KEY_PREFIX}:*", count=100
            )
            if batch:
                await client.delete(*batch)
            if cursor == 0:
                break
    except Exception as exc:  # noqa: BLE001
        logger.warning("bulk mine cache invalidation failed: %s", exc)


async def cache_stats() -> dict[str, Any]:
    """Rough diagnostics for an admin dashboard.

    Counts the number of cached-mine keys currently present. Not a
    hit-rate (that would need INFO CLIENTS-level metrics), just a
    signal that caching is populated.
    """
    client = await get_redis()
    if client is None:
        return {"status": "unavailable"}
    try:
        keys: list[str] = []
        cursor = 0
        while True:
            cursor, batch = await client.scan(
                cursor=cursor, match=f"{_MINE_KEY_PREFIX}:*", count=100
            )
            keys.extend(batch)
            if cursor == 0:
                break
        return {"status": "ok", "mine_keys": len(keys)}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "detail": str(exc)}


__all__ = [
    "cache_stats",
    "get_cached_mine",
    "get_cached_mine_list",
    "invalidate_all_mines",
    "invalidate_mine",
    "set_cached_mine",
    "set_cached_mine_list",
]
