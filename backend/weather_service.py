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
import time
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


# In-process TTL cache keyed on a ~1 km grid cell. The dashboard polls
# /api/telemetry once a second per open tab, and each poll used to make a
# fresh Open-Meteo call -- ~86k requests/day for one tab, well past the
# free tier's 10k/day, after which every frame silently fell back.
# Weather at a mine doesn't change second to second; Open-Meteo's own
# model only updates every 15 minutes.
_CACHE_TTL_SECONDS = 300.0
# A failed fetch is cached too, but briefly, so an offline laptop doesn't
# pay the connect timeout on every poll while still recovering quickly.
_FALLBACK_TTL_SECONDS = 60.0
_cache: dict[tuple[float, float], tuple[float, dict[str, Any]]] = {}


def _cache_key(lat: float, lon: float) -> tuple[float, float]:
    return (round(float(lat), 2), round(float(lon), 2))


def clear_weather_cache() -> None:
    _cache.clear()


def _num(value: Any, default: float) -> float:
    """Open-Meteo reports missing readings as JSON null; treat as default."""
    return default if value is None else float(value)


def _parse_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Shape an Open-Meteo response into the flat schema the app uses."""
    current = data.get("current") or {}
    hourly = data.get("hourly") or {}
    rainfall = _num(current.get("precipitation"), 0.0)
    recent_rain = [v for v in (hourly.get("precipitation") or [])[:6] if v is not None]
    rain_6h_sum = float(sum(recent_rain)) if recent_rain else rainfall
    return {
        "status": "success",
        "source": "Open-Meteo Open API",
        "rainfall_mm": rainfall,
        "humidity_pct": _num(current.get("relative_humidity_2m"), 65.0),
        "temperature_c": _num(current.get("temperature_2m"), 22.0),
        "pressure_hpa": _num(current.get("surface_pressure"), 1013.25),
        "wind_speed_kmh": _num(current.get("wind_speed_10m"), 10.0),
        "rain_rolling_6h": rain_6h_sum,
    }


async def fetch_open_meteo_weather(lat: float, lon: float) -> dict[str, Any]:
    """Fetch current + 6h-antecedent weather for a mine location.

    On any network/HTTP failure returns the fallback payload -- the caller
    (a real-time telemetry endpoint) must never block on the external API,
    so degraded weather data is preferred over 5xx.
    """
    key = _cache_key(lat, lon)
    hit = _cache.get(key)
    if hit is not None and hit[0] > time.monotonic():
        return dict(hit[1])

    # Open-Meteo takes variable lists comma-separated; passing a Python
    # list made httpx send repeated ``current=...&current=...`` params.
    # ``past_hours=6`` + ``forecast_hours=1`` makes the hourly series start
    # six hours ago, so its first six entries really are the antecedent
    # rain. With ``forecast_days=1`` they were 00:00-05:00 local, which
    # is not the last six hours for most of the day.
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": ",".join(
            [
                "precipitation",
                "relative_humidity_2m",
                "temperature_2m",
                "surface_pressure",
                "wind_speed_10m",
            ]
        ),
        "hourly": "precipitation,relative_humidity_2m",
        "past_hours": 6,
        "forecast_hours": 1,
        "timezone": "auto",
    }
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
            response = await client.get(_OPEN_METEO_URL, params=params)
            response.raise_for_status()
            result = _parse_payload(response.json())
        ttl = _CACHE_TTL_SECONDS
    except Exception as exc:  # noqa: BLE001 -- degrade rather than 5xx
        logger.warning(
            "Open-Meteo fetch failed (%s: %s); returning fallback",
            type(exc).__name__,
            exc,
        )
        result = dict(_FALLBACK)
        ttl = _FALLBACK_TTL_SECONDS
    _cache[key] = (time.monotonic() + ttl, result)
    return dict(result)


def fetch_open_meteo_weather_sync(lat: float, lon: float) -> dict[str, Any]:
    """Blocking wrapper for CLI/legacy callers only."""
    return asyncio.run(fetch_open_meteo_weather(lat, lon))


if __name__ == "__main__":
    # Grasberg Mine coordinates
    print("Open-Meteo Response Test:\n", fetch_open_meteo_weather_sync(-4.05, 137.11))
