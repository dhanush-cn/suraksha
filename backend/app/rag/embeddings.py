"""Deterministic ``source_text`` builder for alert embeddings.

The string here is what the embedding model sees, so its content is
what similarity search actually compares. A few properties we want:

* **Every meaningful axis of an alert appears once.** The mine's name
  and company (so "Grasberg" queries hit Grasberg's alerts), the risk
  level and percentage (so "critical" queries rank critical alerts
  higher), the driver (so "pore pressure" queries hit rows where that
  was the top SHAP feature), and the timestamp (so "last week" queries
  have a temporal anchor to correlate against).
* **Stable formatting.** Re-embedding after a model swap should
  produce the same vectors for unchanged rows. String concatenation
  order and formatting are frozen here; changing them requires a
  re-embed of every row.
* **Small.** ~200 tokens tops per alert. Embedding-model context
  limits are generous but a bloated source_text costs latency AND
  makes cosine-distance neighbors less discriminating.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


def build_alert_source_text(
    *,
    mine_name: str,
    company: str,
    risk_level: str,
    risk_percentage: float,
    top_shap_reason: str | None,
    rainfall_mm: float | None,
    pore_pressure_kpa: float | None,
    velocity_mm_h: float | None,
    seismic_rms_g: float | None,
    triggered_at: datetime,
) -> str:
    """Format one alert as the string the embedding model consumes.

    Written as a natural-language paragraph so the embedding model
    (trained on English) produces semantically meaningful vectors.
    JSON-shape input would compress worse and lose "critical" vs
    "warning" as a semantic axis the model already understands.
    """
    parts: list[str] = []
    parts.append(
        f"On {triggered_at:%Y-%m-%d %H:%M} UTC, the {mine_name} mine "
        f"(operated by {company}) recorded a {risk_level.upper()} rockfall risk "
        f"alert at {risk_percentage:.1f}%."
    )
    if top_shap_reason:
        parts.append(f"The top contributing factor was: {top_shap_reason}.")

    sensor_bits: list[str] = []
    if rainfall_mm is not None:
        sensor_bits.append(f"rainfall {rainfall_mm:.1f} mm")
    if pore_pressure_kpa is not None:
        sensor_bits.append(f"pore pressure {pore_pressure_kpa:.1f} kPa")
    if velocity_mm_h is not None:
        sensor_bits.append(f"displacement velocity {velocity_mm_h:.3f} mm/h")
    if seismic_rms_g is not None:
        sensor_bits.append(f"seismic RMS {seismic_rms_g:.4f} g")
    if sensor_bits:
        parts.append("Sensor readings at trigger: " + ", ".join(sensor_bits) + ".")

    return " ".join(parts)


def build_alert_source_text_from_row(alert: Any) -> str:
    """Convenience for callers that have an ``AlertLog`` ORM row +
    the related Mine loaded.

    Kept separate from the kwargs constructor so tests can build a
    canonical source_text without instantiating ORM classes.
    """
    return build_alert_source_text(
        mine_name=alert.mine.name if getattr(alert, "mine", None) else f"Mine #{alert.mine_id}",
        company=alert.mine.company if getattr(alert, "mine", None) else "",
        risk_level=alert.risk_level,
        risk_percentage=alert.risk_percentage,
        top_shap_reason=alert.top_shap_reason,
        rainfall_mm=alert.rainfall_mm,
        pore_pressure_kpa=alert.pore_pressure_kpa,
        velocity_mm_h=alert.velocity_mm_h,
        seismic_rms_g=alert.seismic_rms_g,
        triggered_at=alert.triggered_at,
    )


__all__ = ["build_alert_source_text", "build_alert_source_text_from_row"]
