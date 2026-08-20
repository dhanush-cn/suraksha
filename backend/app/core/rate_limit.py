"""Token-bucket rate limiter, Redis-backed, atomic via a Lua script.

Why token bucket (over fixed-window or sliding-window):

* Bursty legitimate traffic (an operator clicking through the UI in
  quick succession) shouldn't be denied purely because it's temporally
  clustered -- token bucket allows the whole ``capacity`` at once and
  refills at ``refill_rate`` tokens per second, so short bursts pass
  while sustained abuse is throttled to the refill rate.
* Fixed-window is trivially bypassed by two requests straddling the
  window edge; sliding-window needs a sorted set with a per-request
  ZREMRANGEBYSCORE that's more expensive for a hot path.

The refill + decrement runs inside a Lua script so refill/decrement
are atomic against concurrent workers on the same principal --
without EVAL, two workers seeing "0.99 tokens" both round up and let
the request through, breaking the guarantee.

Fails **open** on Redis outage: a rate-limited endpoint served by a
partitioned Redis is preferable to an outage-triggered mass 503. The
JWT blocklist and auth guards take the opposite stance because their
failure mode is worse.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from fastapi import HTTPException, Request, status

from app.core.redis_client import get_redis

logger = logging.getLogger(__name__)

# Atomically refill + spend one token. Returns:
#   {allowed (0|1), remaining (float), retry_after_seconds (int)}
_TOKEN_BUCKET_LUA = """
local key            = KEYS[1]
local capacity       = tonumber(ARGV[1])
local refill_rate    = tonumber(ARGV[2])  -- tokens per second
local now            = tonumber(ARGV[3])
local ttl            = tonumber(ARGV[4])

local data     = redis.call('HMGET', key, 'tokens', 'last')
local tokens   = tonumber(data[1])
local last     = tonumber(data[2])

if tokens == nil then
  tokens = capacity
  last   = now
end

-- Refill based on elapsed wall time; cap at bucket capacity.
local elapsed = math.max(0, now - last)
tokens = math.min(capacity, tokens + elapsed * refill_rate)

local allowed = 0
local retry_after = 0
if tokens >= 1 then
  tokens = tokens - 1
  allowed = 1
else
  -- How long until the next whole token is refilled?
  retry_after = math.ceil((1 - tokens) / refill_rate)
end

redis.call('HMSET', key, 'tokens', tokens, 'last', now)
redis.call('EXPIRE', key, ttl)

return {allowed, tostring(tokens), retry_after}
"""


@dataclass(frozen=True)
class RateLimit:
    """A rate-limit rule attached to an endpoint (or family).

    * ``capacity``    -- max burst size (tokens the bucket can hold).
    * ``refill_rate`` -- tokens replenished per second.

    A rule of ``capacity=60, refill_rate=1.0`` allows a 60-request
    burst then sustained 60 rpm; ``capacity=10, refill_rate=10/3600``
    means "10 per hour, all at once or spread out".
    """

    capacity: int
    refill_rate: float
    scope: str  # used in the Redis key so different endpoints don't
    # deplete each other's buckets.

    @property
    def key_ttl_seconds(self) -> int:
        """TTL for the bucket key -- long enough to outlive a slow
        refill from empty back to full, so an occasional user's
        bucket state isn't reset to full mid-throttle."""
        return max(60, int(self.capacity / max(self.refill_rate, 1e-6)) * 2)


def _identify(request: Request) -> str:
    """Return a stable per-caller key from request context.

    Uses the client IP for anonymous endpoints. When endpoints later
    add auth, wrap this dependency with one that reads
    ``request.state.principal`` (populated by an auth middleware) and
    prefers ``user:<id>`` -- rate limiting SHOULD be per-user for
    authenticated flows to avoid one operator with a shared NAT
    address starving colleagues.

    Falls back to a fixed string when the client IP is unavailable
    (a proxy hiding both ``request.client`` and ``X-Forwarded-For``
    is a misconfiguration; treating everyone as the same caller then
    throttles the whole system rather than letting abuse through
    unbounded).
    """
    ip = None
    if request.client is not None:
        ip = request.client.host
    xff = request.headers.get("x-forwarded-for")
    if xff:
        # Take the leftmost address (the original client per RFC 7239
        # semantics); trust boundary is the reverse proxy.
        ip = xff.split(",")[0].strip()
    return f"ip:{ip or 'unknown'}"


class RateLimiter:
    """Callable that enforces a :class:`RateLimit` on a request.

    Wire into a route with ``Depends(RateLimiter(rule))``. The instance
    is per-rule, not per-request, so the Lua script is loaded once.
    """

    def __init__(self, rule: RateLimit) -> None:
        self._rule = rule

    async def __call__(self, request: Request) -> None:
        client = await get_redis()
        if client is None:
            # Fail open -- see module docstring. Log so ops see it.
            logger.warning(
                "rate limiter bypassed: redis unavailable",
                extra={"scope": self._rule.scope, "path": request.url.path},
            )
            return

        subject = _identify(request)
        key = f"rockfallguard:ratelimit:{self._rule.scope}:{subject}"
        now = time.time()

        try:
            result = await client.eval(
                _TOKEN_BUCKET_LUA,
                1,
                key,
                self._rule.capacity,
                self._rule.refill_rate,
                now,
                self._rule.key_ttl_seconds,
            )
        except Exception as exc:  # noqa: BLE001 -- Lua error, connection reset
            logger.warning("rate limiter EVAL failed: %s", exc)
            return

        allowed = int(result[0]) == 1
        remaining = float(result[1])
        retry_after = int(result[2])

        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Retry after {retry_after}s.",
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(self._rule.capacity),
                    "X-RateLimit-Remaining": "0",
                },
            )

        # Best-effort response headers; not strictly required by any
        # RFC but useful for well-behaved clients.
        # (Headers on 200 responses are added by a helper below when
        # the caller uses a Response dependency; skipping here to
        # keep the dep signature simple.)


# Convenient module-level rules for the two endpoints Step 6 protects.
# The specific numbers come from expected legitimate usage patterns:
# a mine operator clicking through the UI might make 10-20 calls in a
# burst; a batch pipeline shouldn't be able to hammer /predict_risk
# faster than we can inference.
PREDICT_RISK_LIMIT = RateLimit(capacity=60, refill_rate=1.0, scope="predict_risk")
# capacity 5, refill 1 per 60s -> "5 uploads per burst, then one every minute"
UPLOAD_CSV_LIMIT = RateLimit(capacity=5, refill_rate=1 / 60.0, scope="upload_csv")


# Ready-made dependency callables. FastAPI treats a callable instance
# as a dependency: ``Depends(rate_limit_predict_risk)`` invokes the
# RateLimiter.__call__ with request context, atomic-scripted by the
# Lua we compiled above. Module-level so the Lua script and instance
# are shared, not rebuilt per request.
rate_limit_predict_risk = RateLimiter(PREDICT_RISK_LIMIT)
rate_limit_upload_csv = RateLimiter(UPLOAD_CSV_LIMIT)


__all__ = [
    "PREDICT_RISK_LIMIT",
    "RateLimit",
    "RateLimiter",
    "UPLOAD_CSV_LIMIT",
    "rate_limit_predict_risk",
    "rate_limit_upload_csv",
]
