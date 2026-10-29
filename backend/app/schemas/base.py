from __future__ import annotations
from typing import Annotated, Any
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Latitude = Annotated[float, Field(ge=-90.0, le=90.0, description="WGS84 latitude in degrees")]
Longitude = Annotated[float, Field(ge=-180.0, le=180.0, description="WGS84 longitude in degrees")]
Percentage = Annotated[float, Field(ge=0.0, le=100.0)]
PositiveId = Annotated[int, Field(ge=1)]
ShortString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]
MediumString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
PhoneNumber = Annotated[str, StringConstraints(strip_whitespace=True, pattern=r"^\+[1-9]\d{7,14}$")]

class RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True, strict=False, populate_by_name=True, use_enum_values=False)

class ResponseModel(BaseModel):
    model_config = ConfigDict(extra="ignore", from_attributes=True, populate_by_name=True, ser_json_timedelta="float")

class StrictFloatModel(RequestModel):
    model_config = ConfigDict(**RequestModel.model_config, allow_inf_nan=False)

def as_example(*examples: dict[str, Any]) -> dict[str, Any]:
    return {"examples": list(examples)}

__all__ = ["Latitude","Longitude","MediumString","Percentage","PhoneNumber","PositiveId","RequestModel","ResponseModel","ShortString","StrictFloatModel","as_example"]