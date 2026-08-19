from __future__ import annotations
from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated, Self
from pydantic import Field, computed_field, model_validator
from app.schemas.base import Percentage, PositiveId, RequestModel, ResponseModel, StrictFloatModel

Rainfall = Annotated[float, Field(ge=0.0, le=500.0, description="Rainfall (mm)")]
PorePressure = Annotated[float, Field(ge=0.0, le=2_000.0, description="Pore pressure (kPa)")]
Displacement = Annotated[float, Field(ge=-1_000.0, le=100_000.0, description="Cumulative displacement (mm)")]
Velocity = Annotated[float, Field(ge=-100.0, le=10_000.0, description="Velocity (mm/h)")]
Acceleration = Annotated[float, Field(ge=-1_000.0, le=10_000.0, description="Acceleration (mm/h2)")]
SeismicRms = Annotated[float, Field(ge=0.0, le=50.0, description="Seismic RMS (g)")]

class Scenario(StrEnum):
    NORMAL = "normal"; HEAVY_RAIN = "heavy_rain"; MACHINERY_NOISE = "machinery_noise"; CRITICAL_FAILURE = "critical_failure"

class RiskLevel(StrEnum):
    SAFE = "safe"; WARNING = "warning"; CRITICAL = "critical"

class SensorReading(StrictFloatModel):
    mine_id: PositiveId
    rainfall_mm: Rainfall = 0.0
    humidity_pct: Percentage = 50.0
    pore_pressure_kpa: PorePressure = 40.0
    displacement_mm: Displacement = 12.0
    velocity_mm_h: Velocity = 0.05
    acceleration_mm_h2: Acceleration = 0.001
    raw_seismic_rms_g: SeismicRms = 0.02
    rain_rolling_6h: Rainfall | None = Field(default=None, description="Antecedent 6h rainfall; derived if omitted")
    recorded_at: datetime | None = Field(default=None, description="Sensor timestamp (UTC); defaults to receipt time")

    @model_validator(mode="after")
    def _normalise(self) -> Self:
        if self.recorded_at is None:
            object.__setattr__(self, "recorded_at", datetime.now(timezone.utc))
        elif self.recorded_at.tzinfo is None:
            object.__setattr__(self, "recorded_at", self.recorded_at.replace(tzinfo=timezone.utc))
        if self.rain_rolling_6h is None:
            object.__setattr__(self, "rain_rolling_6h", self.rainfall_mm * 2.5)
        elif self.rain_rolling_6h < self.rainfall_mm:
            raise ValueError("rain_rolling_6h must be >= rainfall_mm (6h cumulative cannot be less than the current instantaneous reading)")
        return self

    @computed_field
    @property
    def is_out_of_distribution(self) -> bool:
        return self.velocity_mm_h > 50.0 or self.pore_pressure_kpa > 500.0 or self.rainfall_mm > 200.0 or self.raw_seismic_rms_g > 5.0

class TelemetryQuery(RequestModel):
    scenario: Scenario = Scenario.NORMAL

class ShapExplanation(ResponseModel):
    feature: str
    readable_name: str
    impact_score: float
    direction: str = Field(description="'elevated' or 'stabilising'")
    explanation: str

class ClassProbabilities(ResponseModel):
    safe: Percentage
    warning: Percentage
    critical: Percentage
    @model_validator(mode="after")
    def _check_sums_to_one(self) -> Self:
        total = self.safe + self.warning + self.critical
        if abs(total - 100.0) > 1.0: raise ValueError(f"class probabilities must sum to ~100 (got {total:.2f})")
        return self

class RiskPrediction(ResponseModel):
    risk_percentage: Percentage
    risk_level: RiskLevel
    probabilities: ClassProbabilities
    shap_explanations: list[ShapExplanation] = Field(default_factory=list, max_length=10)
    hydro_kinematic_index: float
    filtered_seismic_g: float
    model_version: str = Field(description="Artifact version that produced this score")
    inference_ms: float = Field(ge=0.0)
    low_confidence: bool = Field(default=False, description="True when the input was out of distribution")
    predicted_at: datetime
    @property
    def top_reason(self) -> str:
        if not self.shap_explanations: return "No dominant driver identified"
        return max(self.shap_explanations, key=lambda item: abs(item.impact_score)).explanation

class PredictionResponse(ResponseModel):
    reading: SensorReading
    prediction: RiskPrediction
    alert_triggered: bool
    alert_threshold_pct: Percentage

__all__ = ["Acceleration","ClassProbabilities","Displacement","PorePressure","PredictionResponse","Rainfall","RiskLevel","RiskPrediction","Scenario","SeismicRms","SensorReading","ShapExplanation","TelemetryQuery","Velocity"]