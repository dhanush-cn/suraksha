"""Emergency-alert Redis Stream (replaces pub/sub for durability).

Why Streams over pub/sub:

* **Pub/sub drops messages when no subscriber is connected.** A
  monitoring service, WebSocket bridge, or downstream analytics that's
  restarting during a critical alert period silently misses events --
  unacceptable for an emergency channel.
* **Streams are durable and replay-able.** ``XADD`` writes to a
  bounded log; consumers replay from a stored offset via
  ``XREADGROUP`` and acknowledge with ``XACK``. A consumer restarting
  or falling behind catches up rather than losing the events it
  missed.
* **Consumer groups let multiple worker replicas share the load.**
  With pub/sub every subscriber sees every message; with a consumer
  group the stream fans out with at-least-once delivery, one message
  to one consumer per group.

Trade-offs called out:

* **Bounded length via ``MAXLEN ~ N``.** A stream that grew forever
  would eventually eat all Redis memory. ``~`` approximates so Redis
  doesn't have to walk the whole stream on every write; exact bound
  isn't worth the perf cost.
* **No cross-datacenter replication.** Redis Streams replicate with
  the standard master-replica pipeline; multi-region durability
  needs a proper broker (Kafka, NATS, cloud pub/sub). For a
  single-region deployment this is fine; documented so an interviewer
  can push on the boundary.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.core.redis_client import get_redis

logger = logging.getLogger(__name__)

STREAM_KEY = "rockfallguard:emergency:events"
DEFAULT_CONSUMER_GROUP = "rockfallguard:emergency:consumers"

# Approximate cap: keep the last ~50k events. At ~1 event / minute /
# mine and 100 mines that's ~35 days of history; enough for a
# post-incident forensic query without unbounded memory growth.
STREAM_MAX_LENGTH = 50_000


async def publish_emergency_event(payload: dict[str, Any]) -> str | None:
    """XADD an emergency-alert event; returns the stream entry id.

    Payload is JSON-encoded into a single ``data`` field so consumers
    parse one string, not a heterogeneous field map that would drift
    across producer versions.

    Returns ``None`` on Redis outage. The caller (:class:`AlertService`)
    treats this as "publish failed, but the DB row was written and the
    dispatch worker was still enqueued" -- the event is recoverable
    from ``alert_logs`` if needed.
    """
    client = await get_redis()
    if client is None:
        logger.warning("emergency stream unavailable: XADD skipped")
        return None
    try:
        entry_id = await client.xadd(
            STREAM_KEY,
            {"data": json.dumps(payload, default=str)},
            maxlen=STREAM_MAX_LENGTH,
            approximate=True,
        )
        return entry_id
    except Exception as exc:  # noqa: BLE001
        logger.warning("XADD to emergency stream failed: %s", exc)
        return None


async def recent_events(*, limit: int = 50) -> list[dict[str, Any]]:
    """Return the newest ``limit`` events, newest first.

    ``XREVRANGE`` (not XRANGE) so the caller gets newest-first without
    a client-side reverse. LIMIT bounds the return so a huge stream
    doesn't hand back megabytes to the ops UI.
    """
    client = await get_redis()
    if client is None:
        return []
    try:
        raw = await client.xrevrange(STREAM_KEY, count=limit)
    except Exception as exc:  # noqa: BLE001
        logger.warning("XREVRANGE failed: %s", exc)
        return []
    events: list[dict[str, Any]] = []
    for entry_id, fields in raw:
        try:
            data = json.loads(fields.get("data", "{}"))
        except (ValueError, TypeError):
            data = {"parse_error": True, "raw": fields.get("data")}
        data["_stream_id"] = entry_id
        events.append(data)
    return events


async def ensure_consumer_group(group: str = DEFAULT_CONSUMER_GROUP) -> None:
    """Idempotently create the consumer group.

    XGROUP CREATE with ``MKSTREAM`` creates the stream too if it
    doesn't exist yet -- so a fresh deployment doesn't need a
    "publish first, subscribe second" ordering dance. Swallows the
    BUSYGROUP error that fires when the group already exists (that's
    the intent, not an error).
    """
    client = await get_redis()
    if client is None:
        return
    try:
        await client.xgroup_create(STREAM_KEY, group, id="$", mkstream=True)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if "BUSYGROUP" in msg:
            return  # already exists
        logger.warning("XGROUP CREATE failed: %s", exc)


async def stream_length() -> int | None:
    """Approximate stream length -- ``XLEN`` is O(1)."""
    client = await get_redis()
    if client is None:
        return None
    try:
        return int(await client.xlen(STREAM_KEY))
    except Exception as exc:  # noqa: BLE001
        logger.warning("XLEN failed: %s", exc)
        return None


__all__ = [
    "DEFAULT_CONSUMER_GROUP",
    "STREAM_KEY",
    "STREAM_MAX_LENGTH",
    "ensure_consumer_group",
    "publish_emergency_event",
    "recent_events",
    "stream_length",
]
