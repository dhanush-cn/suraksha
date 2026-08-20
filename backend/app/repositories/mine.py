"""Mine repository."""

from __future__ import annotations

from typing import Sequence

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ConflictError
from app.db.models import Mine


class MineRepository:
    """CRUD over the ``mines`` table.

    Kept intentionally thin -- no business rules, no HTTP concerns. If a
    caller needs "return all mines a given operator can see" that goes
    in the service layer (Step 3), not here.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list_all(self) -> Sequence[Mine]:
        """All mines ordered by name -- matches the current UI dropdown."""
        result = await self._session.execute(select(Mine).order_by(Mine.name.asc()))
        return result.scalars().all()

    async def get(self, mine_id: int) -> Mine | None:
        return await self._session.get(Mine, mine_id)

    async def create(
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
        """Insert a mine, translating UNIQUE violations into a domain error.

        Raising :class:`ConflictError` here means the route handler gets
        a semantic 409, not a raw sqlite/postgres IntegrityError message
        leaked to the client.
        """
        mine = Mine(
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
        self._session.add(mine)
        try:
            await self._session.flush()
        except IntegrityError as exc:
            await self._session.rollback()
            raise ConflictError(
                message=f"A mine named '{name}' is already registered.",
                internal_detail=str(exc.orig),
            ) from exc
        # Refresh so id / server_default fields are populated on the
        # returned instance without the caller having to remember.
        await self._session.refresh(mine)
        return mine

    async def delete(self, mine_id: int) -> bool:
        """Delete a mine (and, via FK CASCADE, its alert history).

        Returns True if a row was removed, False if the id was not
        present. Not translated to :class:`NotFoundError` -- that call
        belongs to the service so the same "delete missing" outcome can
        map differently for admin cleanup vs a public API.
        """
        result = await self._session.execute(delete(Mine).where(Mine.id == mine_id))
        return (result.rowcount or 0) > 0


__all__ = ["MineRepository"]
