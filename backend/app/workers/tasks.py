"""arq task functions -- CSV scoring, image analysis, emergency dispatch.

Every task is ``async def(ctx, ...)``. arq passes ``ctx`` as the first
argument with keys:

* ``redis``    -- :class:`arq.connections.ArqRedis`, usable for SET NX /
                 LPUSH etc. against the same Redis the queue lives on.
* ``job_try``  -- 1-indexed attempt counter; increments on each Retry.
* ``job_id``, ``enqueue_time``, ``score``, ...

Because the tasks are plain async functions, they can be unit-tested by
constructing a synthetic ``ctx`` dict with an AsyncMock ``redis`` and
awaiting them directly -- no arq worker needed. See
:file:`backend/verify_workers.py` for that pattern.

The dispatch task deliberately owns three properties that the previous
sync ``notification_service.py`` did not:

1. **Retry with exponential backoff** (``arq.Retry`` at 2^attempt seconds).
2. **Idempotency** (Redis ``SET NX`` on ``AlertCreate.idempotency_key``),
   so a duplicate telemetry frame in the same minute bucket does not
   dispatch the same alert twice.
3. **Dead-letter queue** -- after the last retry the failure payload goes
   onto ``rockfallguard:dispatch:dead_letter`` (a bounded LPUSH list),
   so a permanently-broken SMTP/Twilio integration surfaces to ops rather
   than being caught, printed, and dropped.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import smtplib
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

import httpx
from arq import Retry

from app.core.metrics import record_dispatch_outcome
from app.core.config import get_settings

logger = logging.getLogger(__name__)

# --- Dispatch retry policy ---
# 3 attempts, exponential backoff 2^try seconds -> 2s, 4s, 8s.
# Total worst-case dispatch latency: ~14s before dead-letter.
DISPATCH_MAX_TRIES = 3

# --- Redis keys ---
DEAD_LETTER_KEY = "rockfallguard:dispatch:dead_letter"
DEAD_LETTER_MAX_ENTRIES = 1000  # LTRIM ceiling; bounds memory under a
# sustained downstream outage.
IDEMPOTENCY_LOCK_TTL_SECONDS = 3600  # 1 hour: outlives any retry chain
# and gives ops a window to inspect a dispatched key before it expires.


# ---------------------------------------------------------------------------
# 1. CSV batch scoring
# ---------------------------------------------------------------------------


async def score_csv(ctx: dict[str, Any], file_path: str, filename: str) -> dict[str, Any]:
    """Score every row of an uploaded CSV through the risk model.

    The DataFrame iteration + sklearn inference is sync CPU-bound work,
    so it runs in :func:`asyncio.to_thread` -- the arq event loop stays
    free to schedule other queued jobs (image analysis, dispatch)
    concurrently on the same worker.
    """
    return await asyncio.to_thread(_score_csv_sync, file_path, filename)


def _score_csv_sync(file_path: str, filename: str) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    from app.core.config import get_settings

    # Imported lazily: keeps the worker start-up light and avoids paying
    # ml_engine's model-load cost until the first CSV job actually runs.
    from ml_engine import predict_rockfall_risk

    if not os.path.exists(file_path):
        return {"status": "error", "message": f"Uploaded file no longer exists: {file_path}"}

    max_rows = get_settings().max_upload_rows
    df = pd.read_csv(file_path)
    rows, cols = df.shape
    if rows > max_rows:
        # Bounded by Settings.max_upload_rows so a 10-million-row CSV
        # can't monopolise a worker for hours. Returns an error result
        # rather than partial success so the caller decides whether to
        # re-shard and resubmit.
        return {
            "status": "error",
            "message": f"CSV has {rows} rows; the per-upload limit is {max_rows}. Split the file and resubmit.",
            "row_count": rows,
            "limit": max_rows,
        }
    risk_scores: list[float] = []
    safe_count = warn_count = crit_count = 0
    for _, row in df.iterrows():
        pred = predict_rockfall_risk(row.to_dict())
        rp = pred["risk_percentage"]
        risk_scores.append(rp)
        if rp >= 65.0:
            crit_count += 1
        elif rp >= 35.0:
            warn_count += 1
        else:
            safe_count += 1

    avg_risk = float(np.mean(risk_scores)) if risk_scores else 0.0
    max_risk = float(np.max(risk_scores)) if risk_scores else 0.0

    top_driver = "Accelerated Creep Velocity"
    if "pore_pressure_kpa" in df.columns and df["pore_pressure_kpa"].mean() > 50.0:
        top_driver = "Pore Pressure & Hydro-Kinematic Saturation"
    elif "rainfall_mm" in df.columns and df["rainfall_mm"].mean() > 10.0:
        top_driver = "Monsoon Rainfall Intensity"

    return {
        "status": "success",
        "filename": filename,
        "columns": list(df.columns),
        "total_records": int(rows),
        "avg_risk_percentage": round(avg_risk, 1),
        "max_risk_percentage": round(max_risk, 1),
        "safe_records": safe_count,
        "warning_records": warn_count,
        "critical_records": crit_count,
        "top_risk_driver": top_driver,
    }


# ---------------------------------------------------------------------------
# 2. Drone image analysis
# ---------------------------------------------------------------------------


async def analyze_image(ctx: dict[str, Any], image_path: str) -> dict[str, Any]:
    """Run the PyTorch CNN forward pass off the arq event loop."""
    return await asyncio.to_thread(_analyze_image_sync, image_path)


def _analyze_image_sync(image_path: str) -> dict[str, Any]:
    from cv_engine import analyze_drone_pit_image

    if not os.path.exists(image_path):
        return {"status": "error", "message": f"Image file no longer exists: {image_path}"}
    with open(image_path, "rb") as f:
        contents = f.read()
    return analyze_drone_pit_image(contents)


# ---------------------------------------------------------------------------
# 3. Emergency alert dispatch (email + SMS) with retry + idempotency + DLQ
# ---------------------------------------------------------------------------


async def dispatch_alert(ctx: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Deliver an emergency alert over email + SMS.

    Expected payload keys::

        idempotency_key   str  -- AlertCreate.idempotency_key
        mine_id           int
        mine_name         str
        risk_percentage   float
        risk_level        str  ("safe" | "warning" | "critical")
        top_shap_reason   str
        contact_email     str | None
        contact_phone     str | None  (E.164 preferred)

    Returns a JSON-friendly dict with the terminal outcome:
    ``skipped`` (duplicate), ``delivered``, or ``dead_lettered``. A raised
    ``arq.Retry`` schedules the next attempt without returning.
    """
    redis = ctx["redis"]
    attempt = int(ctx.get("job_try", 1))
    idempotency_key = payload["idempotency_key"]
    lock_key = f"rockfallguard:dispatch:lock:{idempotency_key}"

    # SET NX: exactly one worker delivers the alert, even if two identical
    # frames are enqueued from concurrent requests. On retry the lock is
    # already ours (from attempt 1), so the miss-check only meaningfully
    # blocks *new* duplicates -- not our own re-runs.
    if attempt == 1:
        acquired = await redis.set(
            lock_key, "1", ex=IDEMPOTENCY_LOCK_TTL_SECONDS, nx=True
        )
        if not acquired:
            logger.info(
                "dispatch skipped (duplicate idempotency key)",
                extra={"idempotency_key": idempotency_key},
            )
            # Count each channel as skipped so hit-ratio metrics
            # don't drift when a burst of duplicate frames arrives.
            record_dispatch_outcome(channel="email", outcome="skipped")
            record_dispatch_outcome(channel="sms", outcome="skipped")
            return {
                "status": "skipped",
                "reason": "duplicate",
                "idempotency_key": idempotency_key,
            }

    try:
        email_result = await _dispatch_email(payload)
        sms_result = await _dispatch_sms(payload)
    except Exception as exc:  # noqa: BLE001 -- any error must trigger retry/DLQ
        logger.warning("dispatch attempt %d failed: %s", attempt, exc)
        if attempt >= DISPATCH_MAX_TRIES:
            await _dead_letter(redis, payload, exc, attempt)
            # We don't know which channel failed here (dispatch is
            # email-then-sms, first exception aborts), so charge the
            # dead-letter to both. Real-world routing would split into
            # two independent tasks.
            record_dispatch_outcome(channel="email", outcome="dead_lettered")
            record_dispatch_outcome(channel="sms", outcome="dead_lettered")
            return {
                "status": "dead_lettered",
                "attempts": attempt,
                "error": f"{type(exc).__name__}: {exc}",
                "idempotency_key": idempotency_key,
            }
        # Exponential backoff: 2s, 4s, 8s.
        raise Retry(defer=2 ** attempt)

    record_dispatch_outcome(channel="email", outcome=email_result.get("status", "sent"))
    record_dispatch_outcome(channel="sms", outcome=sms_result.get("status", "sent"))
    return {
        "status": "delivered",
        "attempts": attempt,
        "email": email_result,
        "sms": sms_result,
        "idempotency_key": idempotency_key,
    }


async def _dispatch_email(payload: dict[str, Any]) -> dict[str, Any]:
    """Send the SMTP email via a thread pool (smtplib is sync-only)."""
    smtp_host = os.getenv("SMTP_HOST") or os.getenv("SMTP_SERVER")
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASSWORD") or os.getenv("SMTP_PASS")
    to_email = payload.get("contact_email") or "safety@mine.org"
    if not (smtp_host and smtp_user and smtp_pass):
        # No SMTP credentials configured -- treat as a successful simulated
        # send (this is the demo path). Real production would either fail
        # closed or route through a stub SMTP server.
        return {"status": "simulated", "to": to_email, "reason": "SMTP not configured"}

    return await asyncio.to_thread(
        _smtp_send_sync,
        host=smtp_host,
        port=int(os.getenv("SMTP_PORT", "587")),
        user=smtp_user,
        password=smtp_pass,
        to_email=to_email,
        mine_name=str(payload["mine_name"]),
        risk_pct=float(payload["risk_percentage"]),
        risk_level=str(payload["risk_level"]),
        top_reason=str(payload["top_shap_reason"]),
    )


def _smtp_send_sync(
    *,
    host: str,
    port: int,
    user: str,
    password: str,
    to_email: str,
    mine_name: str,
    risk_pct: float,
    risk_level: str,
    top_reason: str,
) -> dict[str, Any]:
    subject = f"CRITICAL SLOPE HAZARD: {mine_name} ({risk_pct:.1f}% Risk)"
    body = (
        f"RockfallGuard emergency dispatch\n\n"
        f"Mine: {mine_name}\n"
        f"Risk: {risk_level.upper()} ({risk_pct:.1f}%)\n"
        f"Driver: {top_reason}\n\n"
        f"MANDATORY EVACUATION: restrict personnel and machinery from lower "
        f"bench sectors immediately."
    )
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to_email
    msg.attach(MIMEText(body, "plain"))
    with smtplib.SMTP(host, port, timeout=10) as server:
        server.starttls()
        server.login(user, password)
        server.sendmail(user, to_email, msg.as_string())
    return {"status": "sent", "to": to_email}


async def _dispatch_sms(payload: dict[str, Any]) -> dict[str, Any]:
    """Send the Twilio SMS via httpx.AsyncClient -- non-blocking."""
    sid = os.getenv("TWILIO_ACCOUNT_SID")
    token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_FROM_NUMBER") or os.getenv("TWILIO_PHONE_NUMBER")
    to_phone = payload.get("contact_phone")
    if not (sid and token and from_number and to_phone):
        return {
            "status": "simulated",
            "to": to_phone,
            "reason": "Twilio not configured or missing recipient",
        }

    url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
    body = (
        f"ROCKFALLGUARD EMERGENCY: {payload['mine_name']} risk "
        f"{float(payload['risk_percentage']):.1f}% ({payload['top_shap_reason']}). "
        f"EVACUATE LOWER BENCH IMMEDIATELY."
    )
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(
            url,
            data={"From": from_number, "To": to_phone, "Body": body},
            auth=(sid, token),
        )
        r.raise_for_status()
    return {"status": "sent", "to": to_phone}


async def _dead_letter(
    redis: Any, payload: dict[str, Any], exc: Exception, attempt: int
) -> None:
    """LPUSH a failure entry into the dead-letter list.

    The list is bounded via LTRIM to :data:`DEAD_LETTER_MAX_ENTRIES` so a
    sustained downstream outage can't consume unbounded Redis memory.
    Ops read with ``LRANGE rockfallguard:dispatch:dead_letter 0 -1``.
    """
    entry = {
        "idempotency_key": payload.get("idempotency_key"),
        "mine_id": payload.get("mine_id"),
        "mine_name": payload.get("mine_name"),
        "risk_percentage": payload.get("risk_percentage"),
        "risk_level": payload.get("risk_level"),
        "top_shap_reason": payload.get("top_shap_reason"),
        "error": f"{type(exc).__name__}: {exc}",
        "attempts": attempt,
        "failed_at": datetime.now(timezone.utc).isoformat(),
    }
    await redis.lpush(DEAD_LETTER_KEY, json.dumps(entry))
    await redis.ltrim(DEAD_LETTER_KEY, 0, DEAD_LETTER_MAX_ENTRIES - 1)
    logger.error(
        "dispatch dead-lettered after %d attempts: %s: %s",
        attempt,
        type(exc).__name__,
        exc,
    )


# ---------------------------------------------------------------------------
# 4. Alert embedding (RAG) -- runs when a new alert lands
# ---------------------------------------------------------------------------


async def embed_alert(ctx: dict[str, Any], alert_id: int) -> dict[str, Any]:
    """Embed one alert's source_text and UPSERT into ``alert_embeddings``.

    * Postgres-only. On SQLite the task returns a "skipped" result so
      the worker's job log is clear about why nothing happened -- the
      RAG feature requires pgvector and the app-level fallbacks handle
      the "not available" UX at query time.
    * Idempotent: rerunning the task overwrites the row rather than
      duplicating. Safe under retry.
    * LLM outage returns a "failed" result; the arq worker doesn't
      retry (max_tries=1 for this function set in WorkerSettings) --
      the embedding is a nice-to-have, not a correctness requirement,
      and thrashing the LLM API on a persistent outage is expensive.
    """
    settings = get_settings()
    if not settings.llm_enabled:
        return {"status": "skipped", "reason": "LLM_API_KEY not configured"}

    # Deferred imports so a SQLite-only run of unrelated worker
    # functions doesn't pay for these (and doesn't import pgvector,
    # which needs a Postgres driver on some code paths).
    from sqlalchemy import text

    from app.db.engine import session_scope
    from app.db.models import AlertLog, Mine
    from app.rag.client import LLMClient
    from app.rag.embeddings import build_alert_source_text_from_row

    async with session_scope() as session:
        if session.bind.dialect.name != "postgresql":  # type: ignore[union-attr]
            return {
                "status": "skipped",
                "reason": f"dialect is {session.bind.dialect.name if session.bind else 'unknown'}; RAG requires postgresql",
            }

        alert = await session.get(AlertLog, alert_id)
        if alert is None:
            return {"status": "error", "message": f"alert {alert_id} not found"}
        # Load mine metadata for the source_text (joinedload isn't
        # needed for a single-row fetch).
        mine = await session.get(Mine, alert.mine_id)
        alert.mine = mine  # attach for the shared source-text builder

        source_text = build_alert_source_text_from_row(alert)
        try:
            embedding = await LLMClient().embed(source_text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("embed_alert failed for alert %s: %s", alert_id, exc)
            return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}

        # UPSERT via ON CONFLICT so re-embedding after a model swap is
        # idempotent -- the alert_id primary key ensures at most one
        # embedding row per alert.
        #
        # Vector encoded as its pgvector string literal (``[1,2,3]``)
        # then CAST -- avoids per-connection codec setup for asyncpg.
        # See app/rag/retrieval.py::_to_vector_literal for the same
        # pattern on the read path.
        from app.rag.retrieval import _to_vector_literal

        await session.execute(
            text(
                """
                INSERT INTO alert_embeddings (alert_id, source_text, model, embedding, created_at)
                VALUES (:alert_id, :source_text, :model, CAST(:embedding AS vector), CURRENT_TIMESTAMP)
                ON CONFLICT (alert_id) DO UPDATE SET
                    source_text = EXCLUDED.source_text,
                    model = EXCLUDED.model,
                    embedding = EXCLUDED.embedding,
                    created_at = EXCLUDED.created_at
                """
            ),
            {
                "alert_id": alert_id,
                "source_text": source_text,
                "model": settings.llm_embedding_model,
                "embedding": _to_vector_literal(embedding),
            },
        )

    return {"status": "embedded", "alert_id": alert_id, "chars": len(source_text)}


__all__ = [
    "DEAD_LETTER_KEY",
    "DEAD_LETTER_MAX_ENTRIES",
    "DISPATCH_MAX_TRIES",
    "IDEMPOTENCY_LOCK_TTL_SECONDS",
    "analyze_image",
    "dispatch_alert",
    "embed_alert",
    "score_csv",
]
