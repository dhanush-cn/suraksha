from __future__ import annotations
import uuid
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from typing import Any, Final
import bcrypt, jwt
from app.core.config import Settings
from app.core.exceptions import TokenExpiredError, TokenInvalidError

LEEWAY_SECONDS: Final[int] = 10
_DUMMY_HASH: Final[bytes] = bcrypt.hashpw(b"dummy-password-for-timing", bcrypt.gensalt(rounds=12))

class TokenType(StrEnum):
    ACCESS = "access"; REFRESH = "refresh"

class Role(StrEnum):
    VIEWER = "viewer"; OPERATOR = "operator"; ADMIN = "admin"
    @property
    def rank(self) -> int: return {Role.VIEWER: 0, Role.OPERATOR: 1, Role.ADMIN: 2}[self]
    def satisfies(self, required: Role) -> bool: return self.rank >= required.rank

def hash_password(password: str, *, rounds: int = 12) -> str:
    if not password: raise ValueError("password must not be empty")
    encoded = password.encode("utf-8")
    if len(encoded) > 72: raise ValueError("password must not exceed 72 bytes when UTF-8 encoded")
    return bcrypt.hashpw(encoded, bcrypt.gensalt(rounds=rounds)).decode("utf-8")

def verify_password(password: str, hashed: str | None) -> bool:
    encoded = password.encode("utf-8")[:72]
    if hashed is None: bcrypt.checkpw(encoded, _DUMMY_HASH); return False
    try: return bcrypt.checkpw(encoded, hashed.encode("utf-8"))
    except ValueError: return False

def needs_rehash(hashed: str, *, rounds: int) -> bool:
    try: current_rounds = int(hashed.split("$")[2])
    except (IndexError, ValueError): return True
    return current_rounds < rounds

def create_token(*, settings: Settings, subject: str, role: Role, token_type: TokenType = TokenType.ACCESS, mine_id: int | None = None, extra_claims: dict[str, Any] | None = None) -> tuple[str, str, datetime]:
    now = datetime.now(timezone.utc)
    ttl = settings.access_token_ttl_seconds if token_type is TokenType.ACCESS else settings.refresh_token_ttl_seconds
    expires_at = now + timedelta(seconds=ttl); jti = uuid.uuid4().hex
    claims: dict[str, Any] = {"sub": subject, "role": str(role), "mine_id": mine_id, "type": str(token_type), "jti": jti, "iat": now, "nbf": now, "exp": expires_at, "iss": settings.jwt_issuer, "aud": settings.jwt_audience}
    if extra_claims:
        reserved = set(claims); claims.update({k: v for k, v in extra_claims.items() if k not in reserved})
    encoded = jwt.encode(claims, settings.jwt_secret.get_secret_value(), algorithm=settings.jwt_algorithm)
    return encoded, jti, expires_at

def decode_token(token: str, *, settings: Settings, expected_type: TokenType = TokenType.ACCESS) -> dict[str, Any]:
    try:
        claims: dict[str, Any] = jwt.decode(token, settings.jwt_secret.get_secret_value(), algorithms=[settings.jwt_algorithm], issuer=settings.jwt_issuer, audience=settings.jwt_audience, leeway=LEEWAY_SECONDS, options={"require": ["exp","iat","nbf","sub","jti","iss","aud"], "verify_exp": True, "verify_nbf": True, "verify_iat": True, "verify_aud": True, "verify_iss": True, "verify_signature": True})
    except jwt.ExpiredSignatureError as exc: raise TokenExpiredError(internal_detail=str(exc)) from exc
    except jwt.InvalidTokenError as exc: raise TokenInvalidError(internal_detail=f"{type(exc).__name__}: {exc}") from exc
    actual_type = claims.get("type")
    if actual_type != str(expected_type): raise TokenInvalidError(internal_detail=f"token type mismatch: expected {expected_type}, got {actual_type}")
    return claims

def revocation_ttl_seconds(claims: dict[str, Any]) -> int:
    exp = claims.get("exp")
    if exp is None: return 0
    remaining = int(exp) - int(datetime.now(timezone.utc).timestamp())
    return max(remaining, 0)

__all__ = ["LEEWAY_SECONDS","Role","TokenType","create_token","decode_token","hash_password","needs_rehash","revocation_ttl_seconds","verify_password"]