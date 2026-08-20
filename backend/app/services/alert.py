"""Alert service -- the one place that decides whether an alert fires.

Collapses the two divergent "should we alert?" branches that used to
live in main.py:

* ``/api/predict_risk`` hardcoded ``if risk_pct >= 60.0`` -- an alert
  fires at 60% regardless of what the mine operator configured.
* ``/api/telemetry`` used ``target_mine.get("alert_threshold_pct", 70.0)``
  -- respected per-mine configuration.

Two endpoints, one sensor reading, two different outcomes depending on
which route saw it. :meth:`should_trigger` is now the only decision;
both handlers route through it, so the mine's configured threshold
always wins.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.streams import publish_emergency_event
from app.db.models import Mine
from app.repositories.alert import AlertRepository
from app.schemas.alert import AlertCreate
from app.schemas.telemetry import RiskLevel
from app.workers.queue import enqueue

# Hard floor: even a mine with no threshold configured (legacy row,
# migration in flight) must never sit above this without alerting. Set
# equal to the historical ``alert_threshold_pct`` column default so
# behaviour is unchanged for correctly-configured mines.
DEFAULT_THRESHOLD_PCT = 70.0


class TelemetryFrame(dict):
    """Type marker for the sensor-frame dict simulator produces.

    Kept as a dict subclass rather than a Pydantic model to avoid
    forcing every caller through validation on the hot path -- the
    simulator's output is already schema-shaped. The alias makes call
    sites read as "this is a telemetry frame, not just any dict".
    """


class AlertService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._repo = AlertRepository(session)

    @staticmethod
    def should_trigger(risk_percentage: float, mine: Mine | None) -> bool:
        """Single source of truth for the alert threshold.

        Uses the mine's configured ``alert_threshold_pct`` when a mine
        is supplied, else falls back to :data:`DEFAULT_THRESHOLD_PCT`
        (matches the historical column default so no configured mine's
        behaviour changes).
        """
        threshold = mine.alert_threshold_pct if mine is not None else DEFAULT_THRESHOLD_PCT
        return risk_percentage >= threshold

    async def record(
        self,
        *,
        mine_id: int,
        prediction: dict[str, Any],
        telemetry: dict[str, Any],
        top_reason: str,
    ) -> None:
        """Append to alert_logs (an audit trail, not a source of state).

        Callers pass ``top_reason`` explicitly so the same string used
        for the dispatch payload appears in the audit row -- keeps
        "what did we tell the operator?" and "what did we log?" in
        sync.
        """
        await self._repo.log(
            mine_id=mine_id,
            risk_percentage=float(prediction["risk_percentage"]),
            risk_level=str(prediction["risk_level"]),
            rainfall_mm=telemetry.get("rainfall_mm"),
            pore_pressure_kpa=telemetry.get("pore_pressure_kpa"),
            velocity_mm_h=telemetry.get("velocity_mm_h"),
            seismic_rms_g=telemetry.get("raw_seismic_rms_g"),
            top_shap_reason=top_reason,
        )

    async def dispatch(
        self,
        *,
        mine: Mine,
        prediction: dict[str, Any],
        top_reason: str,
    ) -> str | None:
        """Enqueue the worker's ``dispatch_alert`` job.

        Uses ``AlertCreate.idempotency_key`` so the worker's ``SET NX``
        lock deduplicates duplicate telemetry frames within the same
        minute bucket. Returns the job id, or ``None`` if the queue is
        unreachable -- the alert audit row is already persisted in
        that case, so ops can re-drive from ``AlertRepository.recent``.
        """
        alert_model = AlertCreate(
            mine_id=mine.id,
            risk_percentage=float(prediction["risk_percentage"]),
            risk_level=RiskLevel(str(prediction["risk_level"]).lower()),
            rainfall_mm=float(prediction.get("rainfall_mm") or 0.0),
            pore_pressure_kpa=float(prediction.get("pore_pressure_kpa") or 0.0),
            velocity_mm_h=float(prediction.get("velocity_mm_h") or 0.0),
            seismic_rms_g=float(prediction.get("seismic_rms_g") or 0.0),
            top_shap_reason=top_reason,
            triggered_at=datetime.now(timezone.utc),
        )
        payload = {
            "idempotency_key": alert_model.idempotency_key,
            "mine_id": mine.id,
            "mine_name": mine.name,
            "risk_percentage": alert_model.risk_percentage,
            "risk_level": prediction["risk_level"],
            "top_shap_reason": top_reason,
            "contact_email": mine.contact_email,
            "contact_phone": mine.contact_phone,
        }
        job_id = await enqueue("dispatch_alert", payload)

        # Emit to the emergency stream too: durable, replay-able,
        # consumed by downstream analytics / WebSocket bridge /
        # external monitoring. Replaces the old pub/sub channel that
        # silently dropped messages when nothing was subscribed.
        # Non-fatal on failure -- the alert row is already in the DB
        # and the worker was still enqueued.
        await publish_emergency_event({**payload, "dispatch_job_id": job_id})
        return job_id


__all__ = ["AlertService", "DEFAULT_THRESHOLD_PCT", "TelemetryFrame"]
