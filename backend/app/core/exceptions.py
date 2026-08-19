from __future__ import annotations
from enum import StrEnum
from typing import Any

class ErrorCode(StrEnum):
    VALIDATION_ERROR = "validation_error"
    INVALID_CREDENTIALS = "invalid_credentials"
    TOKEN_EXPIRED = "token_expired"
    TOKEN_INVALID = "token_invalid"
    TOKEN_REVOKED = "token_revoked"
    AUTHENTICATION_REQUIRED = "authentication_required"
    PERMISSION_DENIED = "permission_denied"
    TENANT_ACCESS_DENIED = "tenant_access_denied"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    PAYLOAD_TOO_LARGE = "payload_too_large"
    UNSUPPORTED_MEDIA_TYPE = "unsupported_media_type"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    INTERNAL_ERROR = "internal_error"
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
    MODEL_NOT_READY = "model_not_ready"

class AppError(Exception):
    status_code: int = 500
    code: ErrorCode = ErrorCode.INTERNAL_ERROR
    message: str = "An unexpected error occurred."
    log_as_error: bool = False
    def __init__(self, message: str | None = None, *, details: dict[str, Any] | None = None, internal_detail: str | None = None, headers: dict[str, str] | None = None) -> None:
        self.message = message or self.__class__.message
        self.details = details or {}
        self.internal_detail = internal_detail
        self.headers = headers or {}
        super().__init__(self.message)

class ValidationAppError(AppError):
    status_code = 422; code = ErrorCode.VALIDATION_ERROR; message = "The request payload failed validation."

class AuthenticationError(AppError):
    status_code = 401; code = ErrorCode.AUTHENTICATION_REQUIRED; message = "Authentication is required to access this resource."
    def __init__(self, message: str | None = None, **kwargs: Any) -> None:
        headers = {"WWW-Authenticate": "Bearer"} | dict(kwargs.pop("headers", {}) or {})
        super().__init__(message, headers=headers, **kwargs)

class InvalidCredentialsError(AuthenticationError):
    code = ErrorCode.INVALID_CREDENTIALS; message = "Incorrect username or password."

class TokenExpiredError(AuthenticationError):
    code = ErrorCode.TOKEN_EXPIRED; message = "Access token has expired."

class TokenInvalidError(AuthenticationError):
    code = ErrorCode.TOKEN_INVALID; message = "Access token is invalid."

class TokenRevokedError(AuthenticationError):
    code = ErrorCode.TOKEN_REVOKED; message = "Access token has been revoked."

class PermissionDeniedError(AppError):
    status_code = 403; code = ErrorCode.PERMISSION_DENIED; message = "You do not have permission to perform this action."

class TenantAccessDeniedError(PermissionDeniedError):
    code = ErrorCode.TENANT_ACCESS_DENIED; message = "You do not have access to this mine."
    def __init__(self, *, mine_id: int, **kwargs: Any) -> None:
        super().__init__(details={"mine_id": mine_id}, internal_detail=f"tenant isolation blocked access to mine_id={mine_id}", **kwargs)

class NotFoundError(AppError):
    status_code = 404; code = ErrorCode.NOT_FOUND; message = "The requested resource was not found."
    def __init__(self, resource: str = "Resource", identifier: Any = None, **kwargs: Any) -> None:
        message = f"{resource} not found."
        details = {"resource": resource}
        if identifier is not None: details["identifier"] = str(identifier)
        super().__init__(message, details=details, **kwargs)

class ConflictError(AppError):
    status_code = 409; code = ErrorCode.CONFLICT; message = "The resource conflicts with an existing one."

class PayloadTooLargeError(AppError):
    status_code = 413; code = ErrorCode.PAYLOAD_TOO_LARGE; message = "Request payload exceeds the permitted size."

class UnsupportedMediaTypeError(AppError):
    status_code = 415; code = ErrorCode.UNSUPPORTED_MEDIA_TYPE; message = "The uploaded file type is not supported."

class RateLimitExceededError(AppError):
    status_code = 429; code = ErrorCode.RATE_LIMIT_EXCEEDED; message = "Rate limit exceeded. Please retry later."
    def __init__(self, *, retry_after_seconds: int, **kwargs: Any) -> None:
        super().__init__(details={"retry_after_seconds": retry_after_seconds}, headers={"Retry-After": str(retry_after_seconds)}, **kwargs)

class DependencyUnavailableError(AppError):
    status_code = 503; code = ErrorCode.DEPENDENCY_UNAVAILABLE; message = "A required downstream service is temporarily unavailable."; log_as_error = True
    def __init__(self, dependency: str, **kwargs: Any) -> None:
        super().__init__(details={"dependency": dependency}, **kwargs)

class ModelNotReadyError(AppError):
    status_code = 503; code = ErrorCode.MODEL_NOT_READY; message = "The risk model is not available. Inference is temporarily disabled."; log_as_error = True

__all__ = ["AppError","AuthenticationError","ConflictError","DependencyUnavailableError","ErrorCode","InvalidCredentialsError","ModelNotReadyError","NotFoundError","PayloadTooLargeError","PermissionDeniedError","RateLimitExceededError","TenantAccessDeniedError","TokenExpiredError","TokenInvalidError","TokenRevokedError","UnsupportedMediaTypeError","ValidationAppError"]