from __future__ import annotations
from datetime import datetime
from typing import Annotated, Self
from pydantic import EmailStr, Field, computed_field, model_validator
from app.schemas.base import Latitude, Longitude, MediumString, Percentage, PhoneNumber, PositiveId, RequestModel, ResponseModel, ShortString

PitDepth = Annotated[float, Field(gt=0.0, le=3_000.0, description="Pit depth (m)")]
SlopeAngle = Annotated[float, Field(gt=0.0, lt=90.0, description="Slope angle (degrees)")]
AlertThreshold = Annotated[float, Field(ge=50.0, le=95.0)]

class MineBase(RequestModel):
    name: ShortString
    company: ShortString
    location_name: MediumString
    latitude: Latitude
    longitude: Longitude
    pit_depth_m: PitDepth = 150.0
    slope_angle_deg: SlopeAngle = 45.0
    contact_email: EmailStr | None = None
    contact_phone: PhoneNumber | None = None
    alert_threshold_pct: AlertThreshold = 70.0
    @model_validator(mode="after")
    def _require_a_contact_channel(self) -> Self:
        if self.contact_email is None and self.contact_phone is None:
            raise ValueError("at least one of contact_email or contact_phone is required so emergency alerts have a destination")
        return self

class MineCreate(MineBase):
    pass

class MineUpdate(RequestModel):
    name: ShortString | None = None
    company: ShortString | None = None
    location_name: MediumString | None = None
    latitude: Latitude | None = None
    longitude: Longitude | None = None
    pit_depth_m: PitDepth | None = None
    slope_angle_deg: SlopeAngle | None = None
    contact_email: EmailStr | None = None
    contact_phone: PhoneNumber | None = None
    alert_threshold_pct: AlertThreshold | None = None
    @model_validator(mode="after")
    def _reject_empty_patch(self) -> Self:
        if not self.model_fields_set: raise ValueError("update payload must contain at least one field")
        return self

class MineSummary(ResponseModel):
    id: PositiveId
    name: str
    company: str
    location_name: str
    latitude: float
    longitude: float
    @computed_field
    @property
    def coordinates(self) -> str:
        return f"{self.latitude:.4f},{self.longitude:.4f}"

class MineDetail(MineSummary):
    pit_depth_m: float
    slope_angle_deg: float
    contact_email: str | None = None
    contact_phone: str | None = None
    alert_threshold_pct: Percentage
    created_at: datetime
    updated_at: datetime | None = None

class MineCreatedResponse(ResponseModel):
    id: PositiveId
    name: str
    message: str = "Mine registered successfully."

__all__ = ["AlertThreshold","MineBase","MineCreate","MineCreatedResponse","MineDetail","MineSummary","MineUpdate","PitDepth","SlopeAngle"]