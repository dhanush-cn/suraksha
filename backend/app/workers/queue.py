"""arq job-queue accessors.

This module is the ONLY place in the app that imports arq directly. Every
endpoint that wants to enqueue work goes through :func:`enqueue`, so if we
later swap arq for something else (RQ, Celery, an SQS-backed shim) the
change is contained here.

Design choices:

* **Graceful Redis degradation.** :func:`get_pool` catches connection
  failures at pool-construction time and returns ``None``. :func:`enqueue`
  and :func:`job_status` propagate that by returning ``None`` / a
  ``status="unavailable"`` payload rather than raising. The rationale: a
  temporary Redis outage should not take down the entire request path,
  and the endpoints already work synchronously for the small-input case.
  Under pytest (where no Redis is running) this makes the auth/tenant
  tests continue to pass without needing a queue.

* **Lazy pool.** ``create_pool`` is deferred until the first enqueue.
  Importing this module never touches the network, so FastAPI startup
  stays fast and the module is safe to import from ``verify_workers.py``
  and other CLI entrypoints.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from urllib.parse import urlparse

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings
from arq.jobs import Job

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# Cached pool + the id() of the event loop it was constructed on.
# Rebuilding on loop change is a no-op in production (uvicorn has one
# process-scoped loop) but essential under starlette TestClient, which
# spins up a fresh loop for each request -- the connections in an
# ArqRedis are bound to their creating loop and raise "Event loop is
# closed" when reused.
_pool: ArqRedis | None = None
_pool_loop_id: int | None = None


def build_redis_settings(*, fail_fast: bool = False) -> RedisSettings:
    """Translate the app-wide ``redis_url`` setting into ``RedisSettings``.

    arq wants host/port/database/password as separate fields; we take a
    single URL to keep parity with :class:`app.core.config.Settings` and
    other clients (celery, cache libraries) that speak DSNs.

    ``fail_fast=True`` bounds ``conn_retries`` to 1 -- used by the API
    enqueue path so a Redis outage returns 503 within ~200 ms instead
    of spending ~5 s in redis-py's default retry loop. The worker
    process wants the opposite (survive a brief blip), so it uses the
    default.
    """
    url = urlparse(str(get_settings().redis_url))
    kwargs: dict[str, Any] = {
        "host": url.hostname or "localhost",
        "port": url.port or 6379,
        "database": int((url.path or "/0").lstrip("/") or 0),
        "password": url.password,
    }
    if fail_fast:
        kwargs["conn_retries"] = 1
        kwargs["conn_retry_delay"] = 0.1
    return RedisSettings(**kwargs)


async def get_pool() -> ArqRedis | None:
    """Return a cached ArqRedis pool, or ``None`` if Redis is unreachable."""
    global _pool, _pool_loop_id
    current_loop_id = id(asyncio.get_running_loop())
    if _pool is not None and _pool_loop_id == current_loop_id:
        return _pool
    if _pool is not None:
        # Stale pool bound to a prior event loop (TestClient re-entrancy).
        # Drop the reference without awaiting aclose(); the old loop is
        # already gone, so the connections will be GC'd on their own.
        logger.debug("resetting stale arq pool from prior event loop")
        _pool = None
        _pool_loop_id = None
    try:
        # API-side enqueue uses fail_fast so a Redis outage returns to
        # the caller quickly instead of piling into redis-py's default
        # 5-attempt retry loop.
        pool = await create_pool(build_redis_settings(fail_fast=True))
        # Ping now so a bad host surfaces here instead of on the first enqueue.
        await pool.ping()
    except Exception as exc:  # noqa: BLE001 -- any failure is "queue down"
        logger.warning("arq pool unavailable: %s", exc)
        return None
    _pool = pool
    _pool_loop_id = current_loop_id
    return _pool


async def close_pool() -> None:
    """Explicitly close the cached pool (for shutdown hooks / tests)."""
    global _pool, _pool_loop_id
    if _pool is not None:
        await _pool.aclose()
        _pool = None
        _pool_loop_id = None


async def enqueue(func_name: str, *args: Any, **kwargs: Any) -> str | None:
    """Enqueue a task and return its ``job_id``, or ``None`` on queue failure.

    ``kwargs`` are forwarded to :meth:`ArqRedis.enqueue_job`; use e.g.
    ``_max_tries=1`` to override per-worker defaults.
    """
    pool = await get_pool()
    if pool is None:
        return None
    try:
        job = await pool.enqueue_job(func_name, *args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("enqueue_job(%s) failed: %s", func_name, exc)
        return None
    return job.job_id if job else None


async def job_status(job_id: str) -> dict[str, Any]:
    """Return a JSON-friendly status snapshot for a job.

    Shape:
        {"job_id", "status", "result"?, "success"?, "enqueue_time"?}

    ``status`` is one of the ``arq.jobs.JobStatus`` string values plus
    ``"unavailable"`` (queue down) or ``"error"`` (introspection failed).
    """
    pool = await get_pool()
    if pool is None:
        return {"job_id": job_id, "status": "unavailable", "detail": "queue unreachable"}
    job = Job(job_id, redis=pool)
    try:
        status = await job.status()
    except Exception as exc:  # noqa: BLE001
        return {"job_id": job_id, "status": "error", "detail": str(exc)}
    # arq's JobStatus is a str-Enum -- use .value so the wire payload is
    # "queued" / "in_progress" / "complete" instead of "JobStatus.queued".
    payload: dict[str, Any] = {"job_id": job_id, "status": status.value}
    try:
        info = await job.result_info()
    except Exception:  # noqa: BLE001 -- info may not exist yet
        info = None
    if info is not None:
        payload["success"] = info.success
        payload["result"] = info.result
        payload["enqueue_time"] = info.enqueue_time.isoformat() if info.enqueue_time else None
        payload["finish_time"] = info.finish_time.isoformat() if info.finish_time else None
    return payload


__all__ = ["build_redis_settings", "close_pool", "enqueue", "get_pool", "job_status"]
