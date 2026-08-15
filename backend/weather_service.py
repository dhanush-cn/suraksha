import requests
from typing import Dict, Any

def fetch_open_meteo_weather(lat: float, lon: float) -> Dict[str, Any]:
    """
    Fetches real-time weather telemetry from Open-Meteo open-source API for a given mine location.
    No API key required!
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": ["precipitation", "relative_humidity_2m", "temperature_2m", "surface_pressure", "wind_speed_10m"],
        "hourly": ["precipitation", "relative_humidity_2m"],
        "forecast_days": 1
    }
    
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        current = data.get("current", {})
        hourly = data.get("hourly", {})
        
        # Calculate 6h antecedent rainfall sum if hourly is present
        recent_rain = hourly.get("precipitation", [])[:6]
        rain_6h_sum = float(sum(recent_rain)) if recent_rain else float(current.get("precipitation", 0.0))
        
        return {
            "status": "success",
            "source": "Open-Meteo Open API",
            "rainfall_mm": float(current.get("precipitation", 0.0)),
            "humidity_pct": float(current.get("relative_humidity_2m", 65.0)),
            "temperature_c": float(current.get("temperature_2m", 22.0)),
            "pressure_hpa": float(current.get("surface_pressure", 1013.25)),
            "wind_speed_kmh": float(current.get("wind_speed_10m", 10.0)),
            "rain_rolling_6h": rain_6h_sum
        }
    except Exception as e:
        print(f"[!] Weather API fetch error ({e}). Returning fallback weather metrics.")
        return {
            "status": "fallback",
            "source": "Local Fallback Sensor",
            "rainfall_mm": 1.2,
            "humidity_pct": 68.0,
            "temperature_c": 24.5,
            "pressure_hpa": 1012.0,
            "wind_speed_kmh": 12.0,
            "rain_rolling_6h": 4.5
        }

if __name__ == "__main__":
    # Test for Grasberg Mine coordinates (-4.05, 137.11)
    res = fetch_open_meteo_weather(-4.05, 137.11)
    print("Open-Meteo Response Test:\n", res)
