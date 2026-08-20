"""Mine service -- CRUD + not-found translation.

The repository returns ``None`` for a missing mine; the service turns
that into a :class:`NotFoundError` so handlers get a semantic error
they can uniformly translate to 404 without re-checking every call.
"""

from __future__ import annotations

from typing import Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundError
from app.db.models import Mine
from app.repositories.mine import MineRepository


class MineService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._repo = MineRepository(session)

    async def list_all(self) -> Sequence[Mine]:
        return await self._repo.list_all()

    async def get_or_404(self, mine_id: int) -> Mine:
        """Fetch a mine by id, raising NotFoundError when missing.

        Two callers use this: the telemetry endpoint (which was falling
        back to ``mines[0]`` when the requested mine didn't exist -- a
        subtle bug that could dispatch alerts against the wrong mine)
        and the delete endpoint. Both want the same behaviour: absent
        mine -> 404, never a silent substitution.
        """
        mine = await self._repo.get(mine_id)
        if mine is None:
            raise NotFoundError(resource="Mine", identifier=mine_id)
        return mine

    async def register(
        self,
        *,
        name: str,
        company: str,
        location_name: str,
        latitude: float,
        longitude: float,
        pit_depth_m: float = 150.0,
        slope_angle_deg: float = 45.0,
        contact_email: str | None = None,
        contact_phone: str | None = None,
        alert_threshold_pct: float = 70.0,
    ) -> Mine:
        return await self._repo.create(
            name=name,
            company=company,
            location_name=location_name,
            latitude=latitude,
            longitude=longitude,
            pit_depth_m=pit_depth_m,
            slope_angle_deg=slope_angle_deg,
            contact_email=contact_email,
            contact_phone=contact_phone,
            alert_threshold_pct=alert_threshold_pct,
        )

    async def delete(self, mine_id: int) -> None:
        """Remove a mine (and its alerts, via FK CASCADE).

        Raises NotFoundError when the id doesn't exist -- consistent
        with ``get_or_404``. The old handler returned 404 with a raw
        detail string; services returning ``AppError`` let every 404
        share one code path.
        """
        deleted = await self._repo.delete(mine_id)
        if not deleted:
            raise NotFoundError(resource="Mine", identifier=mine_id)


__all__ = ["MineService"]
