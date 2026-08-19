from __future__ import annotations
from enum import StrEnum
from typing import Annotated, Any, Generic, TypeVar
from pydantic import Field
from app.schemas.base import RequestModel, ResponseModel

T = TypeVar("T")

class FieldError(ResponseModel):
    field: str = Field(description="Dotted path to the offending field")
    reason: str = Field(description="Human-readable explanation")
    type: str = Field(description="Machine-readable pydantic error type")

class ErrorDetail(ResponseModel):
    code: str = Field(description="Stable machine-readable error code")
    message: str = Field(description="Human-readable, client-safe message")
    details: dict[str, Any] = Field(default_factory=dict)
    correlation_id: str | None = Field(default=None, description="Quote this when reporting the failure")

class ErrorResponse(ResponseModel):
    error: ErrorDetail

class ComponentStatus(StrEnum):
    HEALTHY = "healthy"; DEGRADED = "degraded"; UNAVAILABLE = "unavailable"

class DependencyHealth(ResponseModel):
    name: str
    status: ComponentStatus
    latency_ms: float | None = None
    detail: str | None = None

class HealthResponse(ResponseModel):
    status: ComponentStatus
    version: str
    environment: str
    uptime_seconds: float
    dependencies: list[DependencyHealth] = Field(default_factory=list)
    @property
    def redis_connected(self) -> bool:
        return any(dep.name == "redis" and dep.status is ComponentStatus.HEALTHY for dep in self.dependencies)

class LivenessResponse(ResponseModel):
    status: ComponentStatus = ComponentStatus.HEALTHY

class PaginationParams(RequestModel):
    limit: Annotated[int, Field(ge=1, le=200)] = 50
    offset: Annotated[int, Field(ge=0, le=1_000_000)] = 0

class Page(ResponseModel, Generic[T]):
    items: list[T]
    total: int = Field(ge=0, description="Total rows matching the query")
    limit: int
    offset: int
    @property
    def has_more(self) -> bool:
        return self.offset + len(self.items) < self.total

__all__ = ["ComponentStatus","DependencyHealth","ErrorDetail","ErrorResponse","FieldError","HealthResponse","LivenessResponse","Page","PaginationParams"]