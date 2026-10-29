"""Risk service -- ML inference + explanation extraction.

Wraps ``ml_engine.predict_rockfall_risk`` in a service surface so:

1. There's one place to swap the model implementation (currently
   sklearn + SHAP; later a service call to a dedicated inference
   worker). Handlers don't import ``ml_engine`` directly anymore.
2. The "no SHAP explanations produced -> fall back to this string"
   logic lives in ONE place. Previously two handlers had this branch
   with two different fallback strings ("High Creep Rate" vs
   "Accelerating Displacement Rate"), producing different alert
   audit-log content depending on which endpoint saw the same sensor
   frame.
"""

from __future__ import annotations

from typing import Any

from app.core.metrics import observe_ml_inference

# The ml_engine module lives at backend/ml_engine.py (a sibling of main.py).
# Services import it directly rather than via a repository because the
# model has no persistence surface -- it's stateless inference.
from ml_engine import predict_rockfall_risk

# Single fallback used when SHAP produces zero explanations. Previously
# "High Creep Rate" in one handler, "Accelerating Displacement Rate" in
# the other. Neither was documented as the canonical fallback; the mine
# operator scanning the alert log couldn't tell which was more accurate.
# One string here, referenced from tests, means the alert audit log is
# consistent no matter which entrypoint produced it.
DEFAULT_TOP_REASON = "Accelerating Displacement Rate (SHAP unavailable)"


class RiskService:
    """Stateless -- kept as a class for symmetry with the other services
    and to make it trivially mockable in unit tests."""

    def predict(self, sensor_input: dict[str, Any]) -> dict[str, Any]:
        # Timed at the service edge so the histogram measures
        # "everything the caller sees" -- data prep + inference + SHAP,
        # not just the raw model call.
        with observe_ml_inference():
            return predict_rockfall_risk(sensor_input)

    def extract_top_reason(self, prediction: dict[str, Any]) -> str:
        """First SHAP explanation string, or the shared fallback.

        The old code was ``prediction["shap_explanations"][0]["explanation"]``
        guarded by a truthy check on the list. Same shape here, just
        centralised.
        """
        shap = prediction.get("shap_explanations") or []
        if shap and isinstance(shap[0], dict) and shap[0].get("explanation"):
            return str(shap[0]["explanation"])
        return DEFAULT_TOP_REASON


__all__ = ["DEFAULT_TOP_REASON", "RiskService"]
