"""Open-Meteo weather client.

Async by default (``fetch_open_meteo_weather``) so the endpoints that call
it can serve other requests while waiting on Open-Meteo's ~200-500 ms
response. A sync wrapper (``fetch_open_meteo_weather_sync``) is kept for
the CLI smoke test at the bottom of the file and any legacy caller;
new code should use the async version.

Was previously a blocking ``requests.get`` inside a sync function that
main.py's async endpoints then called -- so every telemetry request
parked its worker thread for the full network round-trip and blocked
the FastAPI event loop from picking up any other request until Open-Meteo
answered.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
_TIMEOUT_SECONDS = 5.0
_FALLBACK: dict[str, Any] = {
    "status": "fallback",
    "source": "Local Fallback Sensor",
    "rainfall_mm": 1.2,
    "humidity_pct": 68.0,
    "temperature_c": 24.5,
    "pressure_hpa": 1012.0,
    "wind_speed_kmh": 12.0,
    "rain_rolling_6h": 4.5,
}


def _parse_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Shape an Open-Meteo response into the flat schema the app uses."""
    current = data.get("current", {})
    hourly = data.get("hourly", {})
    recent_rain = hourly.get("precipitation", [])[:6]
    rain_6h_sum = (
        float(sum(recent_rain))
        if recent_rain
        else float(current.get("precipitation", 0.0))
    )
    return {
        "status": "success",
        "source": "Open-Meteo Open API",
        "rainfall_mm": float(current.get("precipitation", 0.0)),
        "humidity_pct": float(current.get("relative_humidity_2m", 65.0)),
        "temperature_c": float(current.get("temperature_2m", 22.0)),
        "pressure_hpa": float(current.get("surface_pressure", 1013.25)),
        "wind_speed_kmh": float(current.get("wind_speed_10m", 10.0)),
        "rain_rolling_6h": rain_6h_sum,
    }


async def fetch_open_meteo_weather(lat: float, lon: float) -> dict[str, Any]:
    """Fetch current + 6h-antecedent weather for a mine location.

    On any network/HTTP failure returns the fallback payload -- the caller
    (a real-time telemetry endpoint) must never block on the external API,
    so degraded weather data is preferred over 5xx.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": [
            "precipitation",
            "relative_humidity_2m",
            "temperature_2m",
            "surface_pressure",
            "wind_speed_10m",
        ],
        "hourly": ["precipitation", "relative_humidity_2m"],
        "forecast_days": 1,
    }
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
            response = await client.get(_OPEN_METEO_URL, params=params)
            response.raise_for_status()
            return _parse_payload(response.json())
    except Exception as exc:  # noqa: BLE001 -- degrade rather than 5xx
        logger.warning("Open-Meteo fetch failed (%s); returning fallback", exc)
        return dict(_FALLBACK)


def fetch_open_meteo_weather_sync(lat: float, lon: float) -> dict[str, Any]:
    """Blocking wrapper for CLI/legacy callers only."""
    return asyncio.run(fetch_open_meteo_weather(lat, lon))


if __name__ == "__main__":
    # Grasberg Mine coordinates
    print("Open-Meteo Response Test:\n", fetch_open_meteo_weather_sync(-4.05, 137.11))
