"""AlertLog repository."""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.db.models import AlertLog, Mine


class AlertRepository:
    """Read + append against ``alert_logs``.

    No update / delete methods: alerts are an append-only audit log; a
    replayable timeline of what the system saw and did. If ops need to
    "forget" one, that's a schema-level DELETE reviewed separately, not
    a hidden route.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def log(
        self,
        *,
        mine_id: int,
        risk_percentage: float,
        risk_level: str,
        rainfall_mm: float | None,
        pore_pressure_kpa: float | None,
        velocity_mm_h: float | None,
        seismic_rms_g: float | None,
        top_shap_reason: str | None,
        triggered_at: datetime | None = None,
    ) -> AlertLog:
        entry = AlertLog(
            mine_id=mine_id,
            risk_percentage=risk_percentage,
            risk_level=risk_level,
            rainfall_mm=rainfall_mm,
            pore_pressure_kpa=pore_pressure_kpa,
            velocity_mm_h=velocity_mm_h,
            seismic_rms_g=seismic_rms_g,
            top_shap_reason=top_shap_reason,
        )
        # Only set triggered_at explicitly when the caller supplied one;
        # otherwise let the column default (_utcnow) win, so we don't
        # write "None" over a valid default value.
        if triggered_at is not None:
            entry.triggered_at = triggered_at
        self._session.add(entry)
        await self._session.flush()
        await self._session.refresh(entry)
        return entry

    async def recent(self, *, limit: int = 50, mine_id: int | None = None) -> Sequence[AlertLog]:
        """Most recent alerts, newest first.

        ``mine_id`` filter uses the ``ix_alert_logs_mine_triggered``
        composite index; the unfiltered path uses the same index's
        prefix for the ORDER BY. Either way, no full-table scan.

        ``joinedload(Mine)`` fires one JOIN so we don't N+1 the mine
        lookup when the caller wants ``alert.mine.name``.
        """
        stmt = (
            select(AlertLog)
            .options(joinedload(AlertLog.mine))
            .order_by(AlertLog.triggered_at.desc())
            .limit(limit)
        )
        if mine_id is not None:
            stmt = stmt.where(AlertLog.mine_id == mine_id)
        result = await self._session.execute(stmt)
        return result.scalars().all()


__all__ = ["AlertRepository"]
