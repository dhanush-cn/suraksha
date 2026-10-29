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
import time
from typing import Optional

import redis.asyncio as aioredis
from redis.asyncio import Redis

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_client: Redis | None = None
_client_loop_id: int | None = None

# Negative cache. Without it, every caller on every request (rate limiter,
# blocklist, mine cache, stream...) re-dials a Redis that is down and pays
# the connect timeout each time -- a single dashboard poll touches Redis
# four or five times, so a missing Redis turned a 50 ms request into a
# multi-second one. After a failed dial we answer "unavailable" straight
# away for this long, then try again.
_RETRY_AFTER_SECONDS = 5.0
_down_until: float = 0.0


async def get_redis() -> Optional[Redis]:
    """Return the process-wide async Redis client, or ``None`` if unreachable."""
    global _client, _client_loop_id, _down_until
    loop = asyncio.get_running_loop()
    current_loop_id = id(loop)
    if _client is not None and _client_loop_id == current_loop_id:
        return _client
    if _client is not None:
        logger.debug("resetting stale redis client from prior event loop")
        _client = None
        _client_loop_id = None
    if time.monotonic() < _down_until:
        return None

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
        # Log once per back-off window rather than once per call.
        logger.warning(
            "redis unavailable (%s); not retrying for %.0fs", exc, _RETRY_AFTER_SECONDS
        )
        _down_until = time.monotonic() + _RETRY_AFTER_SECONDS
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
    global _client, _client_loop_id, _down_until
    _client = None
    _client_loop_id = None
    _down_until = 0.0


__all__ = ["close_redis", "get_redis", "reset_client_for_tests"]
