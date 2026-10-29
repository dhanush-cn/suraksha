from __future__ import annotations
import hashlib
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Self
from pydantic import Field, computed_field, model_validator
from app.schemas.base import Percentage, PositiveId, RequestModel, ResponseModel
from app.schemas.telemetry import RiskLevel

class DispatchChannel(StrEnum):
    EMAIL = "email"; SMS = "sms"; WEBHOOK = "webhook"

class DispatchStatus(StrEnum):
    PENDING = "pending"; SENT = "sent"; FAILED = "failed"; DEAD_LETTERED = "dead_lettered"; SUPPRESSED = "suppressed"
    @property
    def is_terminal(self) -> bool:
        return self in (DispatchStatus.SENT, DispatchStatus.DEAD_LETTERED, DispatchStatus.SUPPRESSED)

class AlertCreate(RequestModel):
    mine_id: PositiveId
    risk_percentage: Percentage
    risk_level: RiskLevel
    rainfall_mm: float
    pore_pressure_kpa: float
    velocity_mm_h: float
    seismic_rms_g: float
    top_shap_reason: str = Field(max_length=500)
    triggered_at: datetime
    @computed_field
    @property
    def idempotency_key(self) -> str:
        bucket = self.triggered_at.strftime("%Y%m%d%H%M")
        raw = f"{self.mine_id}:{self.risk_level}:{bucket}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

class AlertRead(ResponseModel):
    id: PositiveId
    mine_id: PositiveId
    mine_name: str
    company: str
    risk_percentage: Percentage
    risk_level: RiskLevel
    rainfall_mm: float
    pore_pressure_kpa: float
    velocity_mm_h: float
    seismic_rms_g: float
    top_shap_reason: str
    triggered_at: datetime
    acknowledged_at: datetime | None = None
    acknowledged_by: str | None = None
    @computed_field
    @property
    def is_acknowledged(self) -> bool:
        return self.acknowledged_at is not None

class AlertDispatch(ResponseModel):
    id: PositiveId
    alert_id: PositiveId
    channel: DispatchChannel
    recipient_masked: str
    status: DispatchStatus
    attempt_count: Annotated[int, Field(ge=0, le=20)] = 0
    last_error: str | None = Field(default=None, max_length=500)
    created_at: datetime
    delivered_at: datetime | None = None
    next_retry_at: datetime | None = None
    @model_validator(mode="after")
    def _check_state_consistency(self) -> Self:
        if self.status is DispatchStatus.SENT and self.delivered_at is None:
            raise ValueError("a SENT dispatch must record delivered_at")
        if self.status.is_terminal and self.next_retry_at is not None:
            raise ValueError(f"terminal status '{self.status}' must not schedule a retry")
        return self

class AlertAcknowledge(RequestModel):
    note: str | None = Field(default=None, max_length=500)

__all__ = ["AlertAcknowledge","AlertCreate","AlertDispatch","AlertRead","DispatchChannel","DispatchStatus"]