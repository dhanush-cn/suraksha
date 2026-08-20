"""User repository."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ConflictError
from app.db.models import User


class UserRepository:
    """Login-facing lookups + password rotation.

    All lookups match on ``LOWER(username)`` because usernames are
    case-insensitive in the login flow (``admin`` and ``Admin`` are the
    same account). Storing them lowercased on write would also work but
    would break existing rows on a live-migration; case-folding at read
    time is symmetric and change-resistant.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_username(self, username: str) -> User | None:
        normalised = username.strip().lower()
        result = await self._session.execute(
            select(User).where(func.lower(User.username) == normalised)
        )
        return result.scalar_one_or_none()

    async def get(self, user_id: int) -> User | None:
        return await self._session.get(User, user_id)

    async def create(
        self,
        *,
        username: str,
        password_hash: str,
        role: str,
        mine_id: int | None = None,
        is_active: bool = True,
    ) -> User:
        """Insert a user, mapping UNIQUE(username) collisions to a 409."""
        user = User(
            username=username.strip().lower(),
            password_hash=password_hash,
            role=role,
            mine_id=mine_id,
            is_active=is_active,
        )
        self._session.add(user)
        try:
            await self._session.flush()
        except IntegrityError as exc:
            await self._session.rollback()
            raise ConflictError(
                message=f"A user named '{username}' is already registered.",
                internal_detail=str(exc.orig),
            ) from exc
        await self._session.refresh(user)
        return user

    async def touch_last_login(self, user_id: int, *, when: datetime) -> None:
        """Update ``last_login_at`` without pulling the full row into memory.

        A ``session.get`` -> attribute-set -> flush pattern would fire
        SELECT + UPDATE; this is one UPDATE. Matters on the login hot
        path.
        """
        from sqlalchemy import update

        await self._session.execute(
            update(User).where(User.id == user_id).values(last_login_at=when)
        )

    async def update_password_hash(self, user_id: int, *, password_hash: str) -> None:
        from sqlalchemy import update

        await self._session.execute(
            update(User).where(User.id == user_id).values(password_hash=password_hash)
        )


__all__ = ["UserRepository"]
