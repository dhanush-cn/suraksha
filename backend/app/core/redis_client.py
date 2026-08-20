"""Shared async Redis client.

One :class:`redis.asyncio.Redis` instance per process, cached behind
:func:`get_redis`. All Redis-backed features (rate limiter, cache,
JWT blocklist, emergency stream, dispatch audit list) reach through
this so we don't accumulate a pool per concern.

Design mirrors :mod:`app.workers.queue`:

* **Lazy build.** Constructing the client never touches the network,
  so import stays cheap; the connection opens on first use.
* **Graceful degradation.** A Redis outage returns ``None`` rather
  than raising, and the rate limiter / cache callers handle that as
  "fail open" (rate limit skipped) or "cache miss" (fall through to
  the source of truth). The blocklist and stream fall CLOSED (reject
  requests / return 503) because a silent bypass of revocation would
  be a security regression.
* **Per-event-loop cache.** starlette's TestClient spawns a fresh
  event loop per request; the same trick :mod:`app.workers.queue`
  uses (keyed on ``id(loop)``) applies here.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import redis.asyncio as aioredis
from redis.asyncio import Redis

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_client: Redis | None = None
_client_loop_id: int | None = None


async def get_redis() -> Optional[Redis]:
    """Return the process-wide async Redis client, or ``None`` if unreachable."""
    global _client, _client_loop_id
    current_loop_id = id(asyncio.get_running_loop())
    if _client is not None and _client_loop_id == current_loop_id:
        return _client
    if _client is not None:
        logger.debug("resetting stale redis client from prior event loop")
        _client = None
        _client_loop_id = None

    settings = get_settings()
    try:
        client = aioredis.from_url(
            str(settings.redis_url),
            encoding="utf-8",
            decode_responses=True,
            socket_timeout=settings.redis_socket_timeout_seconds,
            max_connections=settings.redis_max_connections,
            # Bounded retries so a Redis outage returns quickly. The
            # workers path uses the default (5) intentionally -- there
            # the worker WANTS to survive brief blips. The API path
            # wants to fail fast so callers see 503, not a 5-second
            # request hang.
            socket_connect_timeout=1.0,
            retry_on_timeout=False,
        )
        await client.ping()
    except Exception as exc:  # noqa: BLE001 -- any failure is "Redis down"
        logger.warning("redis client unavailable: %s", exc)
        return None
    _client = client
    _client_loop_id = current_loop_id
    return _client


async def close_redis() -> None:
    global _client, _client_loop_id
    if _client is not None:
        await _client.aclose()
        _client = None
        _client_loop_id = None


def reset_client_for_tests() -> None:
    """Drop the cache without awaiting close -- only for tests."""
    global _client, _client_loop_id
    _client = None
    _client_loop_id = None


__all__ = ["close_redis", "get_redis", "reset_client_for_tests"]
