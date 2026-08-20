"""FastAPI dependencies -- session + service factories.

Each request gets a fresh :class:`AsyncSession` via
:func:`get_db_session`; the ``session_scope`` context manager commits
on clean return and rolls back on any raised exception. Services take
the session in their constructor, so a handler that wants ``MineService``
only needs ``Depends(get_mine_service)``.

Kept out of :mod:`main` so tests can override any dep with a fake
without touching module state.
"""

from __future__ import annotations

from typing import AsyncIterator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.engine import session_scope
from app.services import AlertService, AuthService, MineService, RiskService


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """Per-request async session -- one transaction per handler."""
    async with session_scope() as session:
        yield session


async def get_mine_service(
    session: AsyncSession = Depends(get_db_session),
) -> MineService:
    return MineService(session)


async def get_alert_service(
    session: AsyncSession = Depends(get_db_session),
) -> AlertService:
    return AlertService(session)


def get_risk_service() -> RiskService:
    """Stateless -- no session needed."""
    return RiskService()


def get_auth_service() -> AuthService:
    """Wired to the in-process seed roster in :mod:`backend.auth`.

    When the DB-backed swap lands, replace ``_USERS.get`` with a
    ``UserRepository(session).get_by_username`` closure and the rest
    of the app is unchanged.
    """
    from auth import _USERS

    return AuthService(user_lookup=_USERS.get)


__all__ = [
    "get_alert_service",
    "get_auth_service",
    "get_db_session",
    "get_mine_service",
    "get_risk_service",
]
