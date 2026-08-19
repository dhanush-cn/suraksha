from __future__ import annotations
import time
from collections.abc import Awaitable, Callable
from typing import Final
from starlette.datastructures import MutableHeaders
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp
from app.core.config import Settings
from app.core.exceptions import ErrorCode
from app.core.logging import get_logger, new_correlation_id, reset_correlation_id, set_correlation_id

logger = get_logger(__name__)
CORRELATION_HEADER: Final[str] = "X-Request-ID"
_QUIET_PATHS: Final[frozenset[str]] = frozenset({"/health", "/health/live", "/metrics"})

class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        incoming = request.headers.get(CORRELATION_HEADER)
        correlation_id = incoming if incoming and len(incoming) <= 64 and incoming.replace("-", "").isalnum() else new_correlation_id()
        token = set_correlation_id(correlation_id)
        request.state.correlation_id = correlation_id
        started = time.perf_counter()
        try:
            try:
                response = await call_next(request)
            except Exception:
                logger.exception("request failed", extra={"http_method": request.method, "http_path": request.url.path, "duration_ms": round((time.perf_counter() - started) * 1000, 2), "client_ip": _client_ip(request)})
                raise
            duration_ms = (time.perf_counter() - started) * 1000
            response.headers[CORRELATION_HEADER] = correlation_id
            response.headers["X-Response-Time-Ms"] = f"{duration_ms:.2f}"
            if request.url.path not in _QUIET_PATHS:
                logger.info("request completed", extra={"http_method": request.method, "http_path": request.url.path, "http_status": response.status_code, "duration_ms": round(duration_ms, 2), "client_ip": _client_ip(request)})
            return response
        finally:
            reset_correlation_id(token)

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, *, enable_hsts: bool) -> None:
        super().__init__(app); self._enable_hsts = enable_hsts
    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        response = await call_next(request)
        headers = MutableHeaders(scope=None, raw=response.raw_headers)
        headers.setdefault("X-Content-Type-Options", "nosniff")
        headers.setdefault("X-Frame-Options", "DENY")
        headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        headers.setdefault("Permissions-Policy", "geolocation=(), microphone=(), camera=()")
        headers.setdefault("Content-Security-Policy", "default-src 'self'; frame-ancestors 'none'; base-uri 'self'; object-src 'none'")
        if self._enable_hsts: headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
        return response

class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        super().__init__(app); self._max_bytes = max_bytes
    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try: declared = int(content_length)
            except ValueError: return _error_response(status_code=400, code=ErrorCode.VALIDATION_ERROR, message="Malformed Content-Length header.")
            if declared > self._max_bytes:
                logger.warning("request body rejected: too large", extra={"declared_bytes": declared, "limit_bytes": self._max_bytes})
                return _error_response(status_code=413, code=ErrorCode.PAYLOAD_TOO_LARGE, message="Request payload exceeds the permitted size.", details={"limit_bytes": self._max_bytes})
        return await call_next(request)

def _client_ip(request: Request) -> str | None:
    if request.client is not None: return request.client.host
    return None

def _error_response(*, status_code: int, code: ErrorCode, message: str, details: dict[str, object] | None = None) -> JSONResponse:
    from app.core.logging import get_correlation_id
    return JSONResponse(status_code=status_code, content={"error": {"code": str(code), "message": message, "details": details or {}, "correlation_id": get_correlation_id()}})

def install_middleware(app: ASGIApp, settings: Settings) -> None:
    from fastapi import FastAPI
    from starlette.middleware.cors import CORSMiddleware
    from starlette.middleware.gzip import GZipMiddleware
    from starlette.middleware.trustedhost import TrustedHostMiddleware
    assert isinstance(app, FastAPI)
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_request_body_bytes)
    app.add_middleware(SecurityHeadersMiddleware, enable_hsts=settings.environment.is_production)
    app.add_middleware(GZipMiddleware, minimum_size=1_000)
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origin_strings, allow_credentials=settings.cors_allow_credentials, allow_methods=["GET","POST","PATCH","DELETE","OPTIONS"], allow_headers=["Authorization","Content-Type",CORRELATION_HEADER], expose_headers=[CORRELATION_HEADER], max_age=600)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.trusted_hosts)

__all__ = ["CORRELATION_HEADER","BodySizeLimitMiddleware","RequestContextMiddleware","SecurityHeadersMiddleware","install_middleware"]