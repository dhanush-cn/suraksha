"""Mine service -- CRUD + not-found translation.

The repository returns ``None`` for a missing mine; the service turns
that into a :class:`NotFoundError` so handlers get a semantic error
they can uniformly translate to 404 without re-checking every call.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.cache import (
    get_cached_mine,
    get_cached_mine_list,
    invalidate_mine,
    set_cached_mine,
    set_cached_mine_list,
)
from app.core.exceptions import NotFoundError
from app.db.models import Mine
from app.repositories.mine import MineRepository
from app.schemas.auth import TenantScope


def _deserialise_mine(data: dict[str, Any]) -> Mine:
    """dict -> detached Mine ORM instance for cache-hit rehydration.

    Detached (never attached to a session) so touching a relationship
    would raise -- callers of the cache path just read scalar fields.
    Kept explicit rather than using ``Mine(**data)`` so the fromisoformat
    on ``created_at`` matches what set_cached_mine wrote.
    """
    payload = dict(data)
    ts = payload.pop("created_at", None)
    mine = Mine(**payload)
    if ts:
        mine.created_at = datetime.fromisoformat(ts)
    return mine


def _serialise_mine(mine: Mine) -> dict:
    """ORM -> JSON-friendly dict for cache values.

    Kept in the service (not the repo) because caching is a service
    concern -- the repo remains a straight-line SQL translator.
    """
    return {
        "id": mine.id,
        "name": mine.name,
        "company": mine.company,
        "location_name": mine.location_name,
        "latitude": mine.latitude,
        "longitude": mine.longitude,
        "pit_depth_m": mine.pit_depth_m,
        "slope_angle_deg": mine.slope_angle_deg,
        "contact_email": mine.contact_email,
        "contact_phone": mine.contact_phone,
        "alert_threshold_pct": mine.alert_threshold_pct,
        "created_at": mine.created_at.isoformat() if mine.created_at else None,
    }


class MineService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._repo = MineRepository(session)

    async def list_all(self) -> Sequence[Mine]:
        """Unscoped list -- INTERNAL use only (workers, admin cache warmup).

        Route handlers MUST call :meth:`list_visible` with a
        :class:`TenantScope`; this method exists so those internal
        callers don't need to fake a scope. Cached under the
        ``admin`` scope-hash slot so it doubles as the admin
        pre-warm.
        """
        return await self.list_visible(TenantScope(is_admin=True, mine_id=None))

    async def list_visible(self, scope: TenantScope) -> Sequence[Mine]:
        """Cache-first, scope-filtered list of mines the caller may see.

        Two callers, two shapes:

        * Admin -- returns every mine (unfiltered).
        * Operator -- returns exactly one mine (their assigned one) or
          an empty list if the assigned mine no longer exists (mine
          deleted, operator not yet reassigned).

        Cache is per-scope (:meth:`TenantScope.scope_hash`), so the
        two shapes never contaminate each other.
        """
        cached = await get_cached_mine_list(scope.scope_hash())
        if cached is not None:
            return [_deserialise_mine(entry) for entry in cached]

        if scope.is_admin:
            mines = list(await self._repo.list_all())
        elif scope.mine_id is not None:
            one = await self._repo.get(scope.mine_id)
            mines = [one] if one is not None else []
        else:
            # Should never happen -- Principal invariant forbids
            # non-admin without mine_id -- but fail closed if it does.
            mines = []

        await set_cached_mine_list(
            scope.scope_hash(), [_serialise_mine(m) for m in mines]
        )
        return mines

    async def get_or_404(self, mine_id: int) -> Mine:
        """Fetch a mine by id, raising NotFoundError when missing.

        Cache-first: the telemetry endpoint hits this on every request
        (with pit geometry + threshold in scope), so the DB round trip
        was pure repeat work. Invalidation on register / delete keeps
        stale reads bounded to the (short) TTL.

        The old handler silently fell back to ``mines[0]`` when the
        requested mine didn't exist -- a subtle bug that could dispatch
        alerts against the wrong mine. Now a missing mine is always a
        clean NotFoundError.
        """
        cached = await get_cached_mine(mine_id)
        if cached is not None:
            return _deserialise_mine(cached)
        mine = await self._repo.get(mine_id)
        if mine is None:
            raise NotFoundError(resource="Mine", identifier=mine_id)
        await set_cached_mine(mine_id, _serialise_mine(mine))
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
        mine = await self._repo.create(
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
        # Invalidate: the cached list is stale (new mine wasn't in it)
        # and this mine has no cached-single entry yet. Deleting the
        # single-mine key is a no-op cost since it doesn't exist.
        await invalidate_mine(mine.id)
        return mine

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
        # Invalidate AFTER the DELETE succeeds -- otherwise a failed
        # delete would evict a still-valid cache entry and force the
        # next read through the DB unnecessarily.
        await invalidate_mine(mine_id)


__all__ = ["MineService"]
