"""Authentication and authorisation schemas.

Critically, this module removes the guest fallback in ``backend/auth.py``:

    def get_current_user(token = Depends(oauth2_scheme)):
        if not token:
            return {"role": "admin", ...}   # <-- no credentials == full admin

:class:`Principal` can only be constructed from verified token claims. There is
no anonymous variant with elevated privileges, so the bypass cannot be
reintroduced by a later caller forgetting to check.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Any, Self

from pydantic import Field, StringConstraints, model_validator

from app.core.exceptions import TenantAccessDeniedError
from app.core.security import Role, TokenType
from app.schemas.base import PositiveId, RequestModel, ResponseModel


@dataclass(frozen=True)
class TenantScope:
    """Row-level access scope derived from a Principal.

    Every list endpoint takes a ``TenantScope`` (never a raw Principal or a
    plain ``mine_id``) so the "for whom?" question is answered by an object
    the type system knows about, not a convention every author has to
    remember. A repository method whose signature demands ``scope:
    TenantScope`` cannot be called without one, so a new endpoint whose
    author forgot to filter fails at call-site construction, not with a
    silent cross-tenant leak at runtime.

    The dataclass is frozen so it can be passed through async layers
    without concurrent mutation surprises, and small enough to cache-key on
    via :meth:`scope_hash`.
    """

    is_admin: bool
    # Non-admins are guaranteed to have a mine_id by the Principal
    # invariant (a non-admin token with mine_id=None is refused at
    # Principal construction). This field being Optional here is only for
    # the admin case, where mine_id is intentionally None.
    mine_id: int | None

    @classmethod
    def from_principal(cls, principal: "Principal") -> "TenantScope":
        return cls(is_admin=principal.is_admin, mine_id=principal.mine_id)

    def scope_hash(self) -> str:
        """A stable, cache-safe string identifying this scope.

        Used as a suffix on Redis cache keys so an admin's cached list
        of mines can never be served to an operator, and vice versa.
        ``"admin"`` for admins; ``"mine:<id>"`` for operators. Never
        include the username -- two operators of the same mine share
        cached data legitimately, so bucketing by user would just miss.
        """
        return "admin" if self.is_admin else f"mine:{self.mine_id}"

    def visible_mine_ids(self, all_mine_ids: list[int]) -> list[int]:
        """Filter a list of mine ids down to what this scope can see."""
        if self.is_admin:
            return list(all_mine_ids)
        return [mid for mid in all_mine_ids if mid == self.mine_id]

Username = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        to_lower=True,
        min_length=3,
        max_length=64,
        # Alphanumerics plus . _ - only: keeps usernames out of trouble when
        # interpolated into log lines, Redis keys and email headers.
        pattern=r"^[a-z0-9._-]+$",
    ),
]

# Upper bound guards against a bcrypt DoS: hashing a 1 MB "password" is
# expensive, and bcrypt ignores everything past 72 bytes anyway.
Password = Annotated[str, StringConstraints(min_length=12, max_length=128)]


class LoginRequest(RequestModel):
    username: Username
    password: Password


class TokenPair(ResponseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = Field(description="Access token lifetime in seconds")
    expires_at: datetime


class RefreshRequest(RequestModel):
    refresh_token: str = Field(min_length=16, max_length=4_096)


class Principal(ResponseModel):
    """The authenticated caller, built only from verified token claims."""

    user_id: str
    username: str
    role: Role
    mine_id: int | None = Field(
        default=None, description="Tenant scope; None means unscoped (admin)"
    )
    token_id: str = Field(description="jti, for revocation lookups")
    token_type: TokenType = TokenType.ACCESS
    issued_at: datetime
    expires_at: datetime

    @model_validator(mode="after")
    def _check_tenant_scope(self) -> Self:
        """A non-admin must be bound to exactly one mine.

        Without this, a token minted with ``role=operator`` and ``mine_id=None``
        would pass every ``authorize_mine`` check below, because "no scope" reads
        as "unrestricted".
        """
        if self.role is not Role.ADMIN and self.mine_id is None:
            raise ValueError(f"role '{self.role}' requires a mine_id scope")
        return self

    @classmethod
    def from_claims(cls, claims: dict[str, Any]) -> Self:
        """Build from a decoded, signature-verified JWT payload."""
        return cls(
            user_id=str(claims["sub"]),
            username=str(claims.get("username", claims["sub"])),
            role=Role(claims["role"]),
            mine_id=claims.get("mine_id"),
            token_id=str(claims["jti"]),
            token_type=TokenType(claims.get("type", "access")),
            issued_at=datetime.fromtimestamp(claims["iat"], tz=None).astimezone(),
            expires_at=datetime.fromtimestamp(claims["exp"], tz=None).astimezone(),
        )

    @property
    def is_admin(self) -> bool:
        return self.role is Role.ADMIN

    def can_access_mine(self, mine_id: int) -> bool:
        """Tenant isolation check. Admins are unscoped; everyone else is bound."""
        if self.is_admin:
            return True
        return self.mine_id == mine_id

    def authorize_mine(self, mine_id: int) -> None:
        """Raise unless this principal may act on ``mine_id``.

        Prefer this over ``can_access_mine`` at call sites: a bare boolean is
        easy to call and forget to branch on, whereas this fails closed.
        """
        if not self.can_access_mine(mine_id):
            raise TenantAccessDeniedError(mine_id=mine_id)

    def require_role(self, required: Role) -> None:
        """Raise unless this principal meets the required privilege level."""
        from app.core.exceptions import PermissionDeniedError

        if not self.role.satisfies(required):
            raise PermissionDeniedError(
                internal_detail=f"role {self.role} does not satisfy {required}"
            )


class UserRead(ResponseModel):
    """User projection returned by ``/auth/me``. Never includes a password hash."""

    id: PositiveId
    username: str
    role: Role
    mine_id: int | None = None
    company_name: str | None = None
    is_active: bool = True
    last_login_at: datetime | None = None


class LoginResponse(ResponseModel):
    user: UserRead
    tokens: TokenPair


__all__ = [
    "LoginRequest",
    "LoginResponse",
    "Password",
    "Principal",
    "RefreshRequest",
    "TenantScope",
    "TokenPair",
    "UserRead",
    "Username",
]