"""RockfallGuard HTTP API.

This module is now the thin outermost shell of the app:

* env defaults for local dev (production sets the real values)
* FastAPI + CORS + startup
* one exception handler that translates ``AppError`` subclasses raised
  by services into ``{"detail": "..."}`` responses matching the
  existing ``HTTPException`` contract
* route handlers -- each dispatches to a service and shapes the result

Anything that isn't routing lives elsewhere:

* Business rules       -> :mod:`app.services`
* SQL                  -> :mod:`app.repositories`
* HTTP auth guards     -> :mod:`backend.auth` (raises HTTPException on
                          purpose, because auth failures are literally
                          HTTP-layer identity concerns)
* Job enqueueing       -> :mod:`app.workers.queue`

Two behaviour-preserving translations from the previous version:

1. **Alert threshold is one rule.** ``AlertService.should_trigger``
   is the only place. ``/api/predict_risk`` no longer hardcodes 60%;
   both endpoints now respect the mine's configured threshold.

2. **Top-reason fallback is one string.** ``RiskService.extract_top_reason``
   owns the SHAP-empty fallback (was two different strings depending on
   which handler ran).
"""

import os

# app.core.config.Settings requires JWT_SECRET / DATABASE_URL with no
# built-in defaults (by design -- a production deployment must set them
# explicitly). These `setdefault` calls only fill them in for local/dev
# runs that haven't; a real deployment's environment always wins.
os.environ.setdefault(
    "JWT_SECRET",
    "rockfallguard-dev-only-insecure-jwt-secret-do-not-use-in-prod",
)
os.environ.setdefault("DATABASE_URL", "sqlite:///./mines.db")

import json
import shutil
import uuid
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.api.deps import (
    get_alert_service,
    get_auth_service,
    get_mine_service,
    get_risk_service,
)
from app.core.blocklist import blocklist_size, revoke
from app.core.cache import cache_stats
from app.core.exceptions import AppError
from app.core.rate_limit import rate_limit_predict_risk, rate_limit_upload_csv
from app.core.security import decode_token, revocation_ttl_seconds
from app.core.streams import recent_events, stream_length
from app.db.models import Mine
from app.schemas.auth import Principal
from app.services import AlertService, AuthService, MineService, RiskService
from app.workers.queue import enqueue, get_pool, job_status
from app.workers.tasks import DEAD_LETTER_KEY, DEAD_LETTER_MAX_ENTRIES
from auth import (
    enforce_admin_only,
    enforce_tenant_access,
    get_current_principal,
    oauth2_scheme,
)
from app.core.redis_client import get_redis
from database import init_db, get_recent_alerts  # still writes/reads the same SQLite file
from simulator import simulator_instance
from weather_service import fetch_open_meteo_weather

app = FastAPI(
    title="RockfallGuard - Open-Pit Mine Geological Risk Warning API",
    description="Proactive multi-sensor data fusion, weather integration & ML rockfall early warning system.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the database eagerly at import time (not only via the startup
# event): init_db() is idempotent (CREATE TABLE IF NOT EXISTS), and
# starlette's TestClient only runs lifespan/startup handlers when used
# as a context manager. backend/tests/test_app.py instantiates
# TestClient at module scope without one, so relying solely on the
# startup event left the schema uncreated under pytest.
init_db()


@app.on_event("startup")
def on_startup() -> None:
    init_db()


# --- AppError -> HTTPException-shape translator ---------------------------
#
# Services raise ``AppError`` subclasses (NotFoundError, ConflictError,
# PermissionDeniedError, ...). We map them to a JSONResponse with the
# same ``{"detail": "..."}`` shape FastAPI's HTTPException produces --
# tests already assert on this shape and the frontend consumes it, so
# switching to the fancier envelope from app.core.error_handlers would
# be a breaking change for callers.
@app.exception_handler(AppError)
async def _app_error_handler(_request: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message},
        headers=exc.headers or None,
    )


# --- Request/response models used only by this module --------------------


class MineRegistrationRequest(BaseModel):
    name: str = Field(..., example="Grasberg Copper Pit")
    company: str = Field(..., example="Freeport Mining")
    location_name: str = Field(..., example="Sector B Slope")
    latitude: float = Field(..., example=-4.05)
    longitude: float = Field(..., example=137.11)
    pit_depth_m: float = Field(default=350.0)
    slope_angle_deg: float = Field(default=48.0)
    contact_email: Optional[str] = Field(default="safety@mine.org")
    contact_phone: Optional[str] = Field(default="+1-555-0199")
    alert_threshold_pct: float = Field(
        default=70.0, description="Risk threshold (60-80%) to trigger emergency alert"
    )


class TelemetryPredictionRequest(BaseModel):
    mine_id: int
    rainfall_mm: float = 0.0
    humidity_pct: float = 50.0
    pore_pressure_kpa: float = 40.0
    displacement_mm: float = 12.0
    velocity_mm_h: float = 0.05
    acceleration_mm_h2: float = 0.001
    raw_seismic_rms_g: float = 0.02
    rain_rolling_6h: Optional[float] = None


class LoginRequest(BaseModel):
    username: str
    password: str


def _mine_to_dict(mine: Mine) -> Dict[str, Any]:
    """Serialise an ORM Mine to the dict shape the frontend already
    consumes -- same keys as ``database.get_all_mines`` produced."""
    return {
        "id": mine.id,
        "name": mine.name,
        "company": mine.company,
        "location_name": mine.location_name,
        "latitude": mine.latitude,
        "longitude": mine.longitude,
        "pit_depth_m": mine.pit_depth_m,
        "slope_angle_deg": mine.slope_angle_deg,
        "contact_email": mine.contact_email,
        "contact_phone": mine.contact_phone,
        "alert_threshold_pct": mine.alert_threshold_pct,
        "created_at": mine.created_at.isoformat() if mine.created_at else None,
    }


# --- API ENDPOINTS --------------------------------------------------------


@app.get("/api/health")
async def health_check() -> Dict[str, Any]:
    # Uses the async client (same pool everything else uses) instead
    # of maintaining a parallel sync client just for the healthcheck.
    # Absent client -> not connected; short-circuits before ping().
    redis_client = await get_redis()
    return {
        "status": "online",
        "system": "RockfallGuard Proactive Slope Stability Engine",
        "redis_connected": redis_client is not None,
    }


# 0. Authentication
@app.post("/api/auth/login")
def login(req: LoginRequest, auth: AuthService = Depends(get_auth_service)) -> Dict[str, Any]:
    result = auth.login(req.username, req.password)
    return {
        "access_token": result.access_token,
        "token_type": result.token_type,
        "expires_at": result.expires_at,
        "user": {
            "id": result.user_id,
            "username": result.username,
            "role": result.role_display,
            "mine_id": result.mine_id,
            "company_name": result.company_name,
        },
    }


# 0b. Logout -- adds the caller's jti to the Redis blocklist for its
# remaining lifetime. The next request bearing the same token gets 401.
# Uses raw ``oauth2_scheme`` (not get_current_principal) so we still
# reach the revoke path even if the token is already blocklisted --
# double-logout is idempotent, not an error.
@app.post("/api/auth/logout")
async def logout(token: Optional[str] = Depends(oauth2_scheme)) -> Dict[str, Any]:
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    from app.core.config import get_settings
    from app.core.exceptions import TokenExpiredError, TokenInvalidError
    from app.core.security import TokenType

    try:
        claims = decode_token(
            token, settings=get_settings(), expected_type=TokenType.ACCESS
        )
    except (TokenExpiredError, TokenInvalidError):
        # Already unusable; treat as successful revocation.
        return {"status": "revoked"}

    ttl = revocation_ttl_seconds(claims)
    ok = await revoke(claims["jti"], ttl_seconds=ttl)
    if not ok:
        # Redis is down. Failing 200 here would be a lie -- the token
        # is not actually revoked and would keep working elsewhere.
        raise HTTPException(
            status_code=503,
            detail="Revocation service unavailable; token was NOT revoked. Retry when Redis is reachable.",
        )
    return {"status": "revoked", "expires_in_seconds": ttl}


# 1. Mine Registration & Management
@app.post("/api/mines")
async def register_new_mine(
    req: MineRegistrationRequest,
    principal: Principal = Depends(get_current_principal),
    mines: MineService = Depends(get_mine_service),
) -> Dict[str, Any]:
    enforce_admin_only(principal)
    mine = await mines.register(
        name=req.name,
        company=req.company,
        location_name=req.location_name,
        latitude=req.latitude,
        longitude=req.longitude,
        pit_depth_m=req.pit_depth_m,
        slope_angle_deg=req.slope_angle_deg,
        contact_email=req.contact_email,
        contact_phone=req.contact_phone,
        alert_threshold_pct=req.alert_threshold_pct,
    )
    return {
        "status": "success",
        "message": f"Mine '{mine.name}' successfully registered!",
        "mine_id": mine.id,
    }


@app.get("/api/mines")
async def list_mines(mines: MineService = Depends(get_mine_service)) -> List[Dict[str, Any]]:
    return [_mine_to_dict(m) for m in await mines.list_all()]


@app.delete("/api/mines/{mine_id}")
async def remove_mine(
    mine_id: int,
    principal: Principal = Depends(get_current_principal),
    mines: MineService = Depends(get_mine_service),
) -> Dict[str, Any]:
    enforce_admin_only(principal)
    await mines.delete(mine_id)  # raises NotFoundError -> 404 via app handler
    return {"status": "success", "message": f"Mine ID {mine_id} deleted."}


# 2. Real-Time Open-Meteo Weather Integration
@app.get("/api/weather/{lat}/{lon}")
async def get_live_weather(lat: float, lon: float) -> Dict[str, Any]:
    return await fetch_open_meteo_weather(lat, lon)


# 3. Predict Hazard Risk & SHAP Explanations
#
# Used to hardcode ``if risk_pct >= 60.0`` -- diverged from the mine-
# configured threshold. Now routes through AlertService.should_trigger,
# so a single sensor frame produces the same alert decision regardless
# of which endpoint saw it. Rate limit: token bucket, per-IP.
@app.post("/api/predict_risk", dependencies=[Depends(rate_limit_predict_risk)])
async def predict_hazard(
    req: TelemetryPredictionRequest,
    mines: MineService = Depends(get_mine_service),
    alerts: AlertService = Depends(get_alert_service),
    risk: RiskService = Depends(get_risk_service),
) -> Dict[str, Any]:
    input_data = req.model_dump()
    result = risk.predict(input_data)

    # Look the mine up so its configured threshold wins over any hardcode.
    # Fall back to None when the request cites a mine that was deleted
    # since the caller last synced -- we still want to return the risk
    # score, we just can't record/dispatch without a mine.
    try:
        mine = await mines.get_or_404(req.mine_id)
    except Exception:
        mine = None

    if alerts.should_trigger(result["risk_percentage"], mine):
        top_reason = risk.extract_top_reason(result)
        if mine is not None:
            telemetry_view = input_data | {"raw_seismic_rms_g": req.raw_seismic_rms_g}
            await alerts.record(
                mine_id=mine.id,
                prediction=result,
                telemetry=telemetry_view,
                top_reason=top_reason,
            )
    return result


# 4. Stream Combined Live Telemetry + Weather + Risk Prediction
@app.get("/api/telemetry/{mine_id}")
async def get_mine_live_telemetry(
    mine_id: int,
    scenario: str = Query("normal"),
    principal: Principal = Depends(get_current_principal),
    mines: MineService = Depends(get_mine_service),
    alerts: AlertService = Depends(get_alert_service),
    risk: RiskService = Depends(get_risk_service),
) -> Dict[str, Any]:
    enforce_tenant_access(principal, mine_id)

    # get_or_404: the previous handler silently fell back to mines[0]
    # when the requested mine was missing, which could dispatch alerts
    # against the wrong mine. Now a missing mine is a clean 404.
    mine = await mines.get_or_404(mine_id)
    mine_view = _mine_to_dict(mine)

    weather = await fetch_open_meteo_weather(mine.latitude, mine.longitude)
    telemetry = simulator_instance.generate_telemetry_frame(
        scenario=scenario, weather_data=weather, mine_info=mine_view
    )

    prediction = risk.predict({**telemetry, "mine_id": mine_id})

    triggered = alerts.should_trigger(prediction["risk_percentage"], mine)
    dispatch_job_id: Optional[str] = None
    if triggered:
        top_reason = risk.extract_top_reason(prediction)
        await alerts.record(
            mine_id=mine.id,
            prediction=prediction,
            telemetry=telemetry,
            top_reason=top_reason,
        )
        dispatch_job_id = await alerts.dispatch(
            mine=mine, prediction=prediction, top_reason=top_reason
        )

    return {
        "mine": mine_view,
        "weather": weather,
        "telemetry": telemetry,
        "prediction": prediction,
        "alert_triggered": triggered,
        "alert_threshold_pct": mine.alert_threshold_pct,
        "dispatch_job_id": dispatch_job_id,
    }


# 5. Alert History Endpoint
#
# Still reads via the legacy get_recent_alerts because that function
# returns a JOIN-shaped dict the frontend already consumes; migrating
# it to AlertRepository would require the response-shape change to be
# coordinated with the UI.
@app.get("/api/alerts")
def fetch_alerts() -> List[Dict[str, Any]]:
    return get_recent_alerts(limit=50)


# Directory the API uses for uploads that get processed by the worker.
# Worker pods must be able to read this path (shared volume in prod).
_UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "../data/uploads")


# 6. Upload Custom CSV Dataset for Retraining / Validation
#
# Enqueued to the arq worker rather than run inline: a 10k-row upload
# would otherwise freeze the entire event loop until scoring finished.
# Rate limit: 5 uploads per burst, refill 1 / min per IP -- CSV
# scoring is expensive, so the cap is intentionally strict.
@app.post(
    "/api/upload_csv",
    status_code=202,
    dependencies=[Depends(rate_limit_upload_csv)],
)
async def upload_custom_dataset(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    os.makedirs(_UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(_UPLOAD_DIR, f"csv_{uuid.uuid4().hex}.csv")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    job_id = await enqueue("score_csv", file_path, file.filename)
    if job_id is None:
        raise HTTPException(
            status_code=503, detail="Background queue unavailable; try again shortly."
        )
    return {
        "status": "queued",
        "job_id": job_id,
        "poll_url": f"/api/jobs/{job_id}",
        "filename": file.filename,
    }


# 7. Drone UAV Bench Wall PyTorch CNN Image Analysis Endpoint
@app.api_route("/api/analyze_drone_image", methods=["GET", "POST"], status_code=202)
@app.api_route("/api/analyze_drone_image/", methods=["GET", "POST"], status_code=202)
async def analyze_drone_image_endpoint(
    file: Optional[UploadFile] = File(None),
) -> Dict[str, Any]:
    if file is None:
        raise HTTPException(status_code=400, detail="Please select an image file to upload.")
    contents = await file.read()
    if not contents or len(contents) < 10:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    os.makedirs(_UPLOAD_DIR, exist_ok=True)
    image_path = os.path.join(_UPLOAD_DIR, f"img_{uuid.uuid4().hex}.bin")
    with open(image_path, "wb") as buffer:
        buffer.write(contents)

    job_id = await enqueue("analyze_image", image_path)
    if job_id is None:
        raise HTTPException(
            status_code=503, detail="Background queue unavailable; try again shortly."
        )
    return {
        "status": "queued",
        "job_id": job_id,
        "poll_url": f"/api/jobs/{job_id}",
        "filename": file.filename,
    }


# 8. Job Status Polling
@app.get("/api/jobs/{job_id}")
async def get_job_status(
    job_id: str, principal: Principal = Depends(get_current_principal)
) -> Dict[str, Any]:
    return await job_status(job_id)


# 9. Dead-Letter Queue Inspection (admin only)
@app.get("/api/dispatch/dead_letter")
async def get_dispatch_dead_letter(
    limit: int = Query(50, ge=1, le=DEAD_LETTER_MAX_ENTRIES),
    principal: Principal = Depends(get_current_principal),
) -> Dict[str, Any]:
    enforce_admin_only(principal)
    pool = await get_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="Background queue unavailable.")
    raw_entries = await pool.lrange(DEAD_LETTER_KEY, 0, limit - 1)
    entries: List[Dict[str, Any]] = []
    for raw in raw_entries:
        try:
            entries.append(json.loads(raw))
        except (ValueError, TypeError):
            entries.append({"raw": str(raw), "parse_error": True})
    return {"count": len(entries), "entries": entries}


# 10. Emergency Stream Inspection (admin)
#
# Reads recent entries from the Redis Stream that AlertService.dispatch
# XADDs to on every triggered alert. Replaces the old pub/sub channel
# that silently dropped messages when no subscriber was connected.
@app.get("/api/emergency/events")
async def get_emergency_events(
    limit: int = Query(50, ge=1, le=500),
    principal: Principal = Depends(get_current_principal),
) -> Dict[str, Any]:
    enforce_admin_only(principal)
    events = await recent_events(limit=limit)
    return {
        "count": len(events),
        "stream_length": await stream_length(),
        "events": events,
    }


# 11. Cache + Blocklist Diagnostics (admin)
@app.get("/api/ops/diagnostics")
async def get_ops_diagnostics(
    principal: Principal = Depends(get_current_principal),
) -> Dict[str, Any]:
    enforce_admin_only(principal)
    return {
        "cache": await cache_stats(),
        "blocklist_size": await blocklist_size(),
        "emergency_stream_length": await stream_length(),
    }


# Mount Frontend Dashboard Static Directory
frontend_dir = os.path.join(os.path.dirname(__file__), "../frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
