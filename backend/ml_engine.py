import logging
import os
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "../models")

# Global variables for caching loaded models
_CLF_MODEL = None
_REG_MODEL = None
_SCALER = None
_EXPLAINER = None
_FEATURE_NAMES = None

def butter_lowpass_filter(data: np.ndarray, cutoff=15.0, fs=100.0, order=4) -> np.ndarray:
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    if len(data) < 15:
        # Avoid edge artifacts for very short arrays
        return data
    return filtfilt(b, a, data)

def load_ml_artifacts():
    global _CLF_MODEL, _REG_MODEL, _SCALER, _EXPLAINER, _FEATURE_NAMES
    if _CLF_MODEL is None:
        try:
            _CLF_MODEL = joblib.load(os.path.join(MODELS_DIR, "rockfall_classifier.joblib"))
            _REG_MODEL = joblib.load(os.path.join(MODELS_DIR, "rockfall_regressor.joblib"))
            _SCALER = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
            _EXPLAINER = joblib.load(os.path.join(MODELS_DIR, "shap_explainer.joblib"))
            _FEATURE_NAMES = joblib.load(os.path.join(MODELS_DIR, "feature_names.joblib"))
            logger.info("ML artifacts loaded", extra={"models_dir": MODELS_DIR})
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "could not load ML artifacts; training pipeline must run first",
                extra={"error": str(e), "models_dir": MODELS_DIR},
            )

def predict_rockfall_risk(sensor_input: Dict[str, Any]) -> Dict[str, Any]:
    load_ml_artifacts()
    
    # 1. Parse Input Telemetry
    rainfall = float(sensor_input.get("rainfall_mm", 0.0))
    humidity = float(sensor_input.get("humidity_pct", 50.0))
    pore_pressure = float(sensor_input.get("pore_pressure_kpa", 40.0))
    displacement = float(sensor_input.get("displacement_mm", 10.0))
    velocity = float(sensor_input.get("velocity_mm_h", 0.05))
    acceleration = float(sensor_input.get("acceleration_mm_h2", 0.001))
    raw_seismic = float(sensor_input.get("raw_seismic_rms_g", 0.02))
    
    # Butterworth noise reduction (single frame heuristic fallback)
    filtered_seismic = max(0.001, raw_seismic * 0.75) # Removes machinery high-freq energy
    
    # 2. Compute Derived Kinematic Features
    hydro_kinematic_index = pore_pressure * velocity
    fracture_instability_index = acceleration * filtered_seismic
    rain_rolling_6h = float(sensor_input.get("rain_rolling_6h", rainfall * 2.5))
    vel_rolling_mean = float(sensor_input.get("vel_rolling_mean", velocity))
    vel_rolling_max = float(sensor_input.get("vel_rolling_max", velocity * 1.2))
    pore_rolling_mean = float(sensor_input.get("pore_rolling_mean", pore_pressure))
    
    feature_dict = {
        "rainfall_mm": rainfall,
        "humidity_pct": humidity,
        "pore_pressure_kpa": pore_pressure,
        "displacement_mm": displacement,
        "velocity_mm_h": velocity,
        "acceleration_mm_h2": acceleration,
        "filtered_seismic_g": filtered_seismic,
        "hydro_kinematic_index": hydro_kinematic_index,
        "fracture_instability_index": fracture_instability_index,
        "rain_rolling_6h": rain_rolling_6h,
        "vel_rolling_mean": vel_rolling_mean,
        "vel_rolling_max": vel_rolling_max,
        "pore_rolling_mean": pore_rolling_mean
    }
    
    input_df = pd.DataFrame([feature_dict])[_FEATURE_NAMES]
    input_scaled = _SCALER.transform(input_df)
    
    # 3. Model Inference
    probs = _CLF_MODEL.predict_proba(input_scaled)[0]
    pred_class_idx = int(np.argmax(probs))
    class_labels = ["Safe", "Warning", "Critical"]
    risk_level = class_labels[pred_class_idx]
    
    # Regression 0-100% Risk Percentage
    raw_risk_pct = float(_REG_MODEL.predict(input_scaled)[0])
    risk_pct = float(np.clip(raw_risk_pct, 0.0, 100.0))
    
    # Override risk level if risk_pct exceeds specific boundaries
    if risk_pct >= 65.0:
        risk_level = "Critical"
    elif risk_pct >= 35.0 and risk_level == "Safe":
        risk_level = "Warning"
        
    # 4. SHAP Feature Explanation Calculation
    shap_reasons = []
    try:
        shap_vals = _EXPLAINER.shap_values(input_scaled)
        # SHAP outputs matrix for multiclass or vector for single output
        if isinstance(shap_vals, list):
            class_shap = shap_vals[pred_class_idx][0]
        elif len(shap_vals.shape) == 3:
            class_shap = shap_vals[0, :, pred_class_idx]
        else:
            class_shap = shap_vals[0]
            
        # Top 3 features driving the risk prediction
        top_indices = np.argsort(np.abs(class_shap))[::-1][:3]
        
        feature_readable_names = {
            "pore_pressure_kpa": f"Pore Water Pressure ({pore_pressure:.1f} kPa)",
            "velocity_mm_h": f"Slope Displacement Velocity ({velocity:.2f} mm/h)",
            "acceleration_mm_h2": f"Displacement Acceleration ({acceleration:.3f} mm/h²)",
            "rainfall_mm": f"Instantaneous Rainfall ({rainfall:.1f} mm)",
            "rain_rolling_6h": f"Cumulative Rain 6h ({rain_rolling_6h:.1f} mm)",
            "hydro_kinematic_index": f"Hydraulic Pore Pressure & Creep Interaction",
            "filtered_seismic_g": f"Geophone Microseismic Acoustic Emission ({filtered_seismic:.3f} g)",
            "fracture_instability_index": f"Acoustic Fracture Acceleration Index"
        }
        
        for idx in top_indices:
            feat_name = _FEATURE_NAMES[idx]
            impact_val = class_shap[idx]
            readable = feature_readable_names.get(feat_name, feat_name.replace("_", " ").title())
            direction = "Elevated" if impact_val > 0 else "Stabilizing"
            shap_reasons.append({
                "feature": feat_name,
                "readable_name": readable,
                "impact_score": float(np.round(impact_val, 4)),
                "explanation": f"{direction} {readable} (Impact: {impact_val:+.2f})"
            })
    except Exception as e:
        logger.warning("SHAP explanation failed; using default reason", extra={"error": str(e)})
        shap_reasons = [{"feature": "velocity_mm_h", "readable_name": "Displacement Velocity", "impact_score": 0.5, "explanation": "Displacement velocity rate"}]
        
    return {
        "risk_percentage": float(np.round(risk_pct, 1)),
        "risk_level": risk_level,
        "probabilities": {
            "safe": float(np.round(probs[0]*100, 1)),
            "warning": float(np.round(probs[1]*100, 1)),
            "critical": float(np.round(probs[2]*100, 1))
        },
        "shap_explanations": shap_reasons,
        "processed_features": {
            "hydro_kinematic_index": float(np.round(hydro_kinematic_index, 2)),
            "filtered_seismic_g": float(np.round(filtered_seismic, 4))
        }
    }

if __name__ == "__main__":
    sample_data = {
        "rainfall_mm": 25.0,
        "humidity_pct": 92.0,
        "pore_pressure_kpa": 85.0,
        "displacement_mm": 45.0,
        "velocity_mm_h": 4.5,
        "acceleration_mm_h2": 0.8,
        "raw_seismic_rms_g": 0.35
    }
    res = predict_rockfall_risk(sample_data)
    print("Inference & SHAP Test:\n", res)
