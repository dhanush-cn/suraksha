from __future__ import annotations
from typing import Any
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from app.core.config import Settings
from app.core.exceptions import AppError, ErrorCode
from app.core.logging import get_correlation_id, get_logger

logger = get_logger(__name__)

def _envelope(*, code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, "details": details or {}, "correlation_id": get_correlation_id()}}

async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    log = logger.error if exc.log_as_error else logger.warning
    log("application error", extra={"error_code": str(exc.code), "http_status": exc.status_code, "http_path": request.url.path, "internal_detail": exc.internal_detail})
    return JSONResponse(status_code=exc.status_code, content=_envelope(code=str(exc.code), message=exc.message, details=exc.details), headers=exc.headers or None)

async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    field_errors = [{"field": ".".join(str(part) for part in error["loc"][1:]) or "(root)", "reason": error["msg"], "type": error["type"]} for error in exc.errors()]
    logger.info("request validation failed", extra={"http_path": request.url.path, "field_count": len(field_errors)})
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=_envelope(code=str(ErrorCode.VALIDATION_ERROR), message="The request payload failed validation.", details={"fields": field_errors}))

async def response_validation_error_handler(request: Request, exc: ValidationError) -> JSONResponse:
    logger.error("response validation failed — response_model contract violated", exc_info=exc, extra={"http_path": request.url.path})
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=_envelope(code=str(ErrorCode.INTERNAL_ERROR), message="An unexpected error occurred."))

async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    code = {status.HTTP_404_NOT_FOUND: ErrorCode.NOT_FOUND, status.HTTP_401_UNAUTHORIZED: ErrorCode.AUTHENTICATION_REQUIRED, status.HTTP_403_FORBIDDEN: ErrorCode.PERMISSION_DENIED, status.HTTP_413_REQUEST_ENTITY_TOO_LARGE: ErrorCode.PAYLOAD_TOO_LARGE, status.HTTP_429_TOO_MANY_REQUESTS: ErrorCode.RATE_LIMIT_EXCEEDED}.get(exc.status_code, ErrorCode.INTERNAL_ERROR if exc.status_code >= 500 else ErrorCode.VALIDATION_ERROR)
    return JSONResponse(status_code=exc.status_code, content=_envelope(code=str(code), message=str(exc.detail)), headers=getattr(exc, "headers", None))

def build_unhandled_exception_handler(settings: Settings):
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled exception", extra={"http_path": request.url.path, "http_method": request.method})
        details: dict[str, Any] = {}
        if not settings.environment.is_production: details = {"exception_type": type(exc).__name__, "exception_message": str(exc)}
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=_envelope(code=str(ErrorCode.INTERNAL_ERROR), message="An unexpected error occurred. Quote the correlation ID when reporting this.", details=details))
    return unhandled_exception_handler

def install_exception_handlers(app: FastAPI, settings: Settings) -> None:
    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(ValidationError, response_validation_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, build_unhandled_exception_handler(settings))

__all__ = ["app_error_handler","build_unhandled_exception_handler","http_exception_handler","install_exception_handlers","response_validation_error_handler","validation_error_handler"]