"""Authentication service -- credential check + token minting.

The demo seed roster still lives in :mod:`backend.auth` (there's no
user-signup flow yet); this service accepts a lookup callable so tests
and the future DB-backed path can inject their own user store without
subclassing.

FastAPI-layer concerns (``get_current_principal`` dependency,
``enforce_admin_only`` / ``enforce_tenant_access`` HTTP guards) stay
in :mod:`backend.auth` -- those are literally about HTTP identity
extraction and are cleaner as HTTPException-raising helpers than as
service methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from app.core.config import get_settings
from app.core.exceptions import InvalidCredentialsError
from app.core.security import Role, create_token, verify_password


@dataclass(frozen=True)
class LoginResult:
    """What the ``/api/auth/login`` handler shapes into a JSON response."""

    access_token: str
    token_type: str
    expires_at: str
    user_id: int
    username: str
    role_display: str  # "admin" | "user" -- legacy client-facing wording
    mine_id: Optional[int]
    company_name: str


class AuthService:
    """Login flow. Stateless; construct once per request or reuse."""

    def __init__(self, user_lookup: Callable[[str], Any | None]) -> None:
        """`user_lookup` returns a `_SeedUser`-shaped object (id, username,
        password_hash, role, display_role, mine_id, company_name) or None
        for an unknown username. Accepting a callable rather than the
        seed dict directly means the DB-backed swap is a one-line
        change at the composition root."""
        self._lookup = user_lookup

    def login(self, username: str, password: str) -> LoginResult:
        """Verify credentials and mint an access token.

        Runs bcrypt on both the found-user and unknown-user paths (via
        ``verify_password(password, None)`` internally in the security
        module) so a mistyped username can't be distinguished from a
        wrong password by response timing.
        """
        user = self._lookup((username or "").strip().lower())
        if user is None:
            # Still verify against a dummy hash to keep the timing
            # profile symmetric.
            verify_password(password, None)
            raise InvalidCredentialsError()
        if not verify_password(password, user.password_hash):
            raise InvalidCredentialsError()

        settings = get_settings()
        access_token, _jti, expires_at = create_token(
            settings=settings,
            subject=str(user.id),
            role=Role(user.role) if isinstance(user.role, str) else user.role,
            mine_id=user.mine_id,
            extra_claims={"username": user.username},
        )
        return LoginResult(
            access_token=access_token,
            token_type="bearer",
            expires_at=expires_at.isoformat(),
            user_id=user.id,
            username=user.username,
            role_display=user.display_role,
            mine_id=user.mine_id,
            company_name=user.company_name,
        )


__all__ = ["AuthService", "LoginResult"]
