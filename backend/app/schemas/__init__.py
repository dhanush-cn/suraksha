"""Pydantic v2 request/response contracts."""

from app.schemas.base import RequestModel, ResponseModel, StrictFloatModel
from app.schemas.common import ErrorResponse, HealthResponse, Page, PaginationParams

__all__ = [
    "ErrorResponse",
    "HealthResponse",
    "Page",
    "PaginationParams",
    "RequestModel",
    "ResponseModel",
    "StrictFloatModel",
]