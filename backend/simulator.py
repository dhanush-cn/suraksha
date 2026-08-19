import numpy as np
import time
from typing import Dict, Any

class SensorSimulator:
    def __init__(self):
        self.cumulative_disp = 12.4 # mm
        self.last_time = time.time()
        
    def generate_telemetry_frame(
        self,
        scenario: str = "normal",
        weather_data: Dict[str, Any] = None,
        mine_info: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Generates realistic 1-second interval sensor telemetry based on chosen operational scenario.

        Args:
            scenario: One of "normal", "heavy_rain", "machinery_noise", "critical_failure".
            weather_data: Live weather payload used to drive rainfall/humidity inputs.
            mine_info: Registered mine record. Steeper slopes and deeper pits are
                       inherently less stable, so pit geometry scales the baseline
                       pore pressure and creep velocity.
        """
        # Default base weather if not provided
        if not weather_data:
            weather_data = {"rainfall_mm": 0.5, "humidity_pct": 55.0, "rain_rolling_6h": 1.2}
            
        rain = weather_data.get("rainfall_mm", 0.5)
        humidity = weather_data.get("humidity_pct", 55.0)
        rain_6h = weather_data.get("rain_rolling_6h", rain * 2.0)
        
        if scenario == "normal":
            velocity = np.random.uniform(0.01, 0.08) # mm/h
            acceleration = np.random.normal(0.001, 0.0005)
            pore_pressure = 38.0 + rain_6h * 1.5 + np.random.normal(0, 1.0)
            raw_seismic = np.abs(np.random.normal(0.015, 0.005))
            
        elif scenario == "heavy_rain":
            rain = float(np.clip(rain + np.random.uniform(15.0, 35.0), 10.0, 90.0))
            rain_6h = float(rain_6h + rain * 0.8)
            humidity = float(np.clip(humidity + 35.0, 75.0, 99.0))
            velocity = np.random.uniform(0.4, 1.1)
            acceleration = np.random.normal(0.05, 0.02)
            pore_pressure = 65.0 + rain_6h * 2.2 + np.random.normal(0, 2.0)
            raw_seismic = np.abs(np.random.normal(0.08, 0.02))
            
        elif scenario == "machinery_noise":
            velocity = np.random.uniform(0.05, 0.12)
            acceleration = np.random.normal(0.005, 0.002)
            pore_pressure = 42.0 + np.random.normal(0, 1.0)
            # High amplitude raw seismic noise from hauling trucks (0.25-0.45g)
            raw_seismic = np.abs(np.random.normal(0.35, 0.08))
            
        elif scenario == "critical_failure":
            rain = float(np.clip(rain + 25.0, 20.0, 110.0))
            rain_6h = float(rain_6h + 40.0)
            velocity = np.random.uniform(3.5, 9.8) # Rapid creep velocity (mm/h)
            acceleration = np.random.uniform(0.45, 1.8) # High acceleration spike
            pore_pressure = 95.0 + np.random.uniform(15.0, 35.0) # High pore pressure (kPa)
            raw_seismic = np.abs(np.random.normal(0.48, 0.10)) # Severe microseismic cracking
        else:
            velocity = 0.05
            acceleration = 0.001
            pore_pressure = 40.0
            raw_seismic = 0.02

        # Scale by pit geometry when a registered mine is supplied. A 60-degree
        # slope is materially less stable than a 30-degree one, so a shared
        # baseline across every mine would be physically wrong.
        # Applied BEFORE displacement accumulates, otherwise the reported
        # velocity would not be the one that produced the reported displacement.
        if mine_info:
            slope_angle = float(mine_info.get("slope_angle_deg") or 45.0)
            pit_depth = float(mine_info.get("pit_depth_m") or 150.0)
            # Normalised around the 45-degree / 150 m reference case.
            slope_factor = max(0.5, min(2.0, slope_angle / 45.0))
            depth_factor = max(0.8, min(1.6, 1.0 + (pit_depth - 150.0) / 1500.0))
            pore_pressure *= slope_factor
            velocity *= slope_factor * depth_factor

        # Increment displacement
        self.cumulative_disp += (velocity * (1.0 / 3600.0)) # mm increment per sec

        return {
            "scenario": scenario,
            "rainfall_mm": float(np.round(rain, 2)),
            "humidity_pct": float(np.round(humidity, 1)),
            "pore_pressure_kpa": float(np.round(pore_pressure, 2)),
            "displacement_mm": float(np.round(self.cumulative_disp, 3)),
            "velocity_mm_h": float(np.round(velocity, 4)),
            "acceleration_mm_h2": float(np.round(acceleration, 4)),
            "raw_seismic_rms_g": float(np.round(raw_seismic, 4)),
            "rain_rolling_6h": float(np.round(rain_6h, 2)),
            "timestamp": time.strftime("%H:%M:%S")
        }

simulator_instance = SensorSimulator()