"""OAuth2/JWT authentication wiring for the RockfallGuard API.

This module used to carry a hand-rolled HMAC JWT implementation *and* a
guest fallback that handed out an admin ``Principal`` whenever a request
arrived with no bearer token at all:

    def get_current_user(token = Depends(oauth2_scheme)):
        if not token:
            return {"id": 0, "role": "admin", ...}   # <-- no credentials == full admin

That fallback is gone. There is no anonymous identity in this module; every
protected route requires a verified token, and ``get_current_principal``
raises 401 when one is missing or invalid. Signing, verification, hashing
and role semantics now all live in ``app.core.security`` (PyJWT, pinned
algorithm, bcrypt) instead of being reimplemented here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from app.core.config import get_settings
from app.core.exceptions import TokenExpiredError, TokenInvalidError
from app.core.security import Role, TokenType, create_token, decode_token, hash_password, verify_password
from app.schemas.auth import Principal

# Standard OAuth2 Bearer Token Scheme (points to /api/auth/login).
# auto_error=False so we can raise our own 401 with a clear message instead
# of FastAPI's generic "Not authenticated".
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)


# --------------------------------------------------------------------------
# Demo credential store.
#
# RockfallGuard has no user-signup flow yet, so authentication is backed by
# a small fixed roster rather than a users table. Each entry keeps the
# *legacy* display role ("admin" / "user") that the rest of the app and its
# tests speak, separate from the `app.core.security.Role` used internally
# for token claims and tenant-scoping decisions -- "user" isn't a member of
# that enum, it maps onto Role.OPERATOR, a tenant-scoped role bound to a
# single mine_id.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _SeedUser:
    id: int
    username: str
    password_hash: str
    role: Role
    display_role: str
    mine_id: Optional[int]
    company_name: str


def _seed_users() -> dict[str, _SeedUser]:
    return {
        "admin": _SeedUser(
            id=1,
            username="admin",
            password_hash=hash_password("admin123"),
            role=Role.ADMIN,
            display_role="admin",
            mine_id=None,
            company_name="Global Mining Admin",
        ),
        "grasberg_user": _SeedUser(
            id=2,
            username="grasberg_user",
            password_hash=hash_password("user123"),
            role=Role.OPERATOR,
            display_role="user",
            mine_id=1,
            company_name="Freeport Copper-Gold",
        ),
    }


# Hashed once at import time; bcrypt is deliberately slow so this shouldn't
# run per-request.
_USERS: dict[str, _SeedUser] = _seed_users()


def authenticate_user(username: str, password: str) -> _SeedUser:
    """Verify credentials against the seed roster.

    Runs bcrypt on both the found-user and unknown-user paths (via
    ``verify_password(..., None)``) so a mistyped username can't be
    distinguished from a wrong password by response timing.
    """
    user = _USERS.get((username or "").strip().lower())
    if user is None:
        verify_password(password, None)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
        )
    if not verify_password(password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
        )
    return user


def issue_login_tokens(user: _SeedUser) -> dict:
    """Mint an access token for an authenticated seed user and shape the
    OAuth2-style login response the frontend/tests expect."""
    settings = get_settings()
    access_token, _jti, expires_at = create_token(
        settings=settings,
        subject=str(user.id),
        role=user.role,
        mine_id=user.mine_id,
        extra_claims={"username": user.username},
    )
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_at": expires_at.isoformat(),
        "user": {
            "id": user.id,
            "username": user.username,
            "role": user.display_role,
            "mine_id": user.mine_id,
            "company_name": user.company_name,
        },
    }


# --------------------------------------------------------------------------
# FastAPI dependency: resolves the authenticated caller from a verified JWT.
# --------------------------------------------------------------------------


def get_current_principal(token: Optional[str] = Depends(oauth2_scheme)) -> Principal:
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated. A Bearer token is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    settings = get_settings()
    try:
        claims = decode_token(token, settings=settings, expected_type=TokenType.ACCESS)
    except (TokenExpiredError, TokenInvalidError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=exc.message,
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    return Principal.from_claims(claims)


# RBAC / tenant-isolation helpers. These raise plain HTTPExceptions (rather
# than the app.core.exceptions.AppError hierarchy) so the response shape
# stays the flat `{"detail": "..."}` the existing API and its tests use.


def enforce_tenant_access(principal: Principal, target_mine_id: int) -> None:
    """Tenant isolation: admins are unscoped, everyone else must match mine_id."""
    if not principal.can_access_mine(target_mine_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"Tenant Access Denied: User '{principal.username}' is restricted "
                f"to Mine ID {principal.mine_id} and cannot access Mine ID {target_mine_id}."
            ),
        )


def enforce_admin_only(principal: Principal) -> None:
    if not principal.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin Privilege Required: Only Admin users can perform this operation.",
        )
