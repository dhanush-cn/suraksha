import os

# app.core.config.Settings requires JWT_SECRET / DATABASE_URL with no
# built-in defaults (by design -- a production deployment must set them
# explicitly). These `setdefault` calls only fill them in for local/dev
# runs that haven't; a real deployment's environment always wins.
os.environ.setdefault("JWT_SECRET", "rockfallguard-dev-only-insecure-jwt-secret-do-not-use-in-prod")
os.environ.setdefault("DATABASE_URL", "sqlite:///./mines.db")

import json
import shutil
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from database import init_db, register_mine, get_all_mines, delete_mine, log_alert, get_recent_alerts
from weather_service import fetch_open_meteo_weather
from ml_engine import predict_rockfall_risk
from simulator import simulator_instance
from redis_service import redis_service
from app.schemas.auth import Principal
from app.schemas.alert import AlertCreate
from app.schemas.telemetry import RiskLevel
from app.workers.queue import enqueue, get_pool, job_status
from app.workers.tasks import DEAD_LETTER_KEY, DEAD_LETTER_MAX_ENTRIES
from auth import (
    authenticate_user,
    enforce_admin_only,
    enforce_tenant_access,
    get_current_principal,
    issue_login_tokens,
)

app = FastAPI(
    title="RockfallGuard - Open-Pit Mine Geological Risk Warning API",
    description="Proactive multi-sensor data fusion, weather integration & ML rockfall early warning system.",
    version="2.0.0"
)

# Enable CORS for local web interface access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the database eagerly at import time (not only via the startup
# event): init_db() is idempotent (CREATE TABLE IF NOT EXISTS), and
# starlette's TestClient only runs lifespan/startup handlers when used as a
# context manager (`with TestClient(app) as client`). backend/tests/test_app.py
# instantiates TestClient at module scope without one, so relying solely on
# the startup event left the schema uncreated under pytest.
init_db()

@app.on_event("startup")
def on_startup():
    init_db()

# Pydantic Schemas
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
    alert_threshold_pct: float = Field(default=70.0, description="Risk threshold (60-80%) to trigger emergency alert")

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

# --- API ENDPOINTS ---

@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "system": "RockfallGuard Proactive Slope Stability Engine",
        "redis_connected": redis_service.is_connected(),
    }

# 0. Authentication
@app.post("/api/auth/login")
def login(req: LoginRequest):
    user = authenticate_user(req.username, req.password)
    return issue_login_tokens(user)

# 1. Mine Registration & Management
@app.post("/api/mines")
def register_new_mine(req: MineRegistrationRequest, principal: Principal = Depends(get_current_principal)):
    enforce_admin_only(principal)
    try:
        mine_id = register_mine(
            name=req.name,
            company=req.company,
            location_name=req.location_name,
            latitude=req.latitude,
            longitude=req.longitude,
            pit_depth_m=req.pit_depth_m,
            slope_angle_deg=req.slope_angle_deg,
            contact_email=req.contact_email,
            contact_phone=req.contact_phone,
            alert_threshold_pct=req.alert_threshold_pct
        )
        return {"status": "success", "message": f"Mine '{req.name}' successfully registered!", "mine_id": mine_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/api/mines")
def list_mines():
    return get_all_mines()

@app.delete("/api/mines/{mine_id}")
def remove_mine(mine_id: int, principal: Principal = Depends(get_current_principal)):
    enforce_admin_only(principal)
    deleted = delete_mine(mine_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Mine ID {mine_id} not found.")
    return {"status": "success", "message": f"Mine ID {mine_id} deleted."}

# 2. Real-Time Open-Meteo Weather Integration
@app.get("/api/weather/{lat}/{lon}")
async def get_live_weather(lat: float, lon: float):
    return await fetch_open_meteo_weather(lat, lon)

# 3. Predict Hazard Risk & SHAP Explanations
@app.post("/api/predict_risk")
def predict_hazard(req: TelemetryPredictionRequest):
    input_data = req.dict()
    result = predict_rockfall_risk(input_data)
    
    # Check if risk exceeds threshold to auto-log alert
    risk_pct = result["risk_percentage"]
    if risk_pct >= 60.0:
        top_reason = result["shap_explanations"][0]["explanation"] if result["shap_explanations"] else "High Creep Rate"
        log_alert(
            mine_id=req.mine_id,
            risk_percentage=risk_pct,
            risk_level=result["risk_level"],
            rainfall_mm=req.rainfall_mm,
            pore_pressure_kpa=req.pore_pressure_kpa,
            velocity_mm_h=req.velocity_mm_h,
            seismic_rms_g=req.raw_seismic_rms_g,
            top_shap_reason=top_reason
        )
    return result

# 4. Stream Combined Live Telemetry + Weather + Risk Prediction
@app.get("/api/telemetry/{mine_id}")
async def get_mine_live_telemetry(mine_id: int, scenario: str = Query("normal"), principal: Principal = Depends(get_current_principal)):
    enforce_tenant_access(principal, mine_id)
    mines = get_all_mines()
    target_mine = next((m for m in mines if m["id"] == mine_id), None)
    if not target_mine:
        target_mine = mines[0] if mines else {
            "id": 1, "name": "Default Open Pit Mine", "latitude": -4.05, "longitude": 137.11, "alert_threshold_pct": 70.0
        }

    # Fetch real live weather for mine GPS coordinates (async httpx: does
    # not block the event loop while Open-Meteo answers)
    weather = await fetch_open_meteo_weather(target_mine["latitude"], target_mine["longitude"])

    # Generate sensor telemetry frame for specific mine
    telemetry = simulator_instance.generate_telemetry_frame(scenario=scenario, weather_data=weather, mine_info=target_mine)

    # Run ML Inference
    prediction_input = {**telemetry, "mine_id": mine_id}
    prediction = predict_rockfall_risk(prediction_input)

    # Auto log alert if risk exceeds mine threshold (e.g. >60-80%)
    alert_threshold = target_mine.get("alert_threshold_pct", 70.0)
    risk_pct = prediction["risk_percentage"]
    is_alert_triggered = risk_pct >= alert_threshold

    dispatch_job_id: Optional[str] = None
    if is_alert_triggered:
        top_reason = prediction["shap_explanations"][0]["explanation"] if prediction["shap_explanations"] else "Accelerating Displacement Rate"
        log_alert(
            mine_id=mine_id,
            risk_percentage=risk_pct,
            risk_level=prediction["risk_level"],
            rainfall_mm=telemetry["rainfall_mm"],
            pore_pressure_kpa=telemetry["pore_pressure_kpa"],
            velocity_mm_h=telemetry["velocity_mm_h"],
            seismic_rms_g=telemetry["raw_seismic_rms_g"],
            top_shap_reason=top_reason
        )

        # Enqueue emergency dispatch (email + SMS) to the worker. Derive
        # the idempotency_key via AlertCreate so re-triggering the same
        # alert within one minute (e.g., duplicate telemetry frames)
        # dedupes at the worker via Redis SET NX. If the queue is down,
        # enqueue() returns None -- the alert is still persisted in the
        # DB, and ops can re-drive from get_recent_alerts().
        alert_model = AlertCreate(
            mine_id=mine_id,
            risk_percentage=risk_pct,
            risk_level=RiskLevel(prediction["risk_level"].lower()),
            rainfall_mm=telemetry["rainfall_mm"],
            pore_pressure_kpa=telemetry["pore_pressure_kpa"],
            velocity_mm_h=telemetry["velocity_mm_h"],
            seismic_rms_g=telemetry["raw_seismic_rms_g"],
            top_shap_reason=top_reason,
            triggered_at=datetime.now(timezone.utc),
        )
        dispatch_payload = {
            "idempotency_key": alert_model.idempotency_key,
            "mine_id": mine_id,
            "mine_name": target_mine.get("name", f"Mine #{mine_id}"),
            "risk_percentage": risk_pct,
            "risk_level": prediction["risk_level"],
            "top_shap_reason": top_reason,
            "contact_email": target_mine.get("contact_email"),
            "contact_phone": target_mine.get("contact_phone"),
        }
        dispatch_job_id = await enqueue("dispatch_alert", dispatch_payload)

    return {
        "mine": target_mine,
        "weather": weather,
        "telemetry": telemetry,
        "prediction": prediction,
        "alert_triggered": is_alert_triggered,
        "alert_threshold_pct": alert_threshold,
        "dispatch_job_id": dispatch_job_id,
    }

# 5. Alert History Endpoint
@app.get("/api/alerts")
def fetch_alerts():
    return get_recent_alerts(limit=50)

# Directory the API uses for uploads that get processed by the worker.
# Worker pods must be able to read this path (shared volume in prod).
_UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "../data/uploads")

# 6. Upload Custom CSV Dataset for Retraining / Validation
#
# Enqueued to the arq worker rather than run inline: the previous
# implementation ran df.iterrows() through predict_rockfall_risk
# synchronously inside an async endpoint, so a 10k-row upload froze the
# entire FastAPI event loop until it finished. Now the endpoint returns
# 202 with a job_id the client polls at GET /api/jobs/{job_id}.
@app.post("/api/upload_csv", status_code=202)
async def upload_custom_dataset(file: UploadFile = File(...)):
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    os.makedirs(_UPLOAD_DIR, exist_ok=True)
    # Per-upload filename: prevents two concurrent uploads from clobbering
    # each other, and lets the worker read the exact bytes it was queued
    # for even if another upload lands in the meantime.
    file_path = os.path.join(_UPLOAD_DIR, f"csv_{uuid.uuid4().hex}.csv")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # max_tries=1 for score_csv is set per-function in WorkerSettings;
    # CSV scoring is deterministic and retrying a corrupt input just
    # wastes worker capacity.
    job_id = await enqueue("score_csv", file_path, file.filename)
    if job_id is None:
        raise HTTPException(status_code=503, detail="Background queue unavailable; try again shortly.")
    return {
        "status": "queued",
        "job_id": job_id,
        "poll_url": f"/api/jobs/{job_id}",
        "filename": file.filename,
    }

# 7. Drone UAV Bench Wall PyTorch CNN Image Analysis Endpoint
#
# Same rationale as /api/upload_csv: the torch forward pass is CPU-bound
# work that must not run in the request path.
@app.api_route("/api/analyze_drone_image", methods=["GET", "POST"], status_code=202)
@app.api_route("/api/analyze_drone_image/", methods=["GET", "POST"], status_code=202)
async def analyze_drone_image_endpoint(file: Optional[UploadFile] = File(None)):
    if file is None:
        raise HTTPException(status_code=400, detail="Please select an image file to upload.")
    contents = await file.read()
    if not contents or len(contents) < 10:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    os.makedirs(_UPLOAD_DIR, exist_ok=True)
    image_path = os.path.join(_UPLOAD_DIR, f"img_{uuid.uuid4().hex}.bin")
    with open(image_path, "wb") as buffer:
        buffer.write(contents)

    # max_tries=1 for analyze_image is set per-function in WorkerSettings.
    job_id = await enqueue("analyze_image", image_path)
    if job_id is None:
        raise HTTPException(status_code=503, detail="Background queue unavailable; try again shortly.")
    return {
        "status": "queued",
        "job_id": job_id,
        "poll_url": f"/api/jobs/{job_id}",
        "filename": file.filename,
    }

# 8. Job Status Polling
#
# Clients hit this to check whether a background job (score_csv,
# analyze_image, dispatch_alert) has completed and to read its result.
@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str, principal: Principal = Depends(get_current_principal)):
    return await job_status(job_id)

# 9. Dead-Letter Queue Inspection (admin only)
#
# The dispatch worker LPUSHes failed alert deliveries here after the
# retry cap. Ops can inspect and re-drive, so nothing gets silently
# dropped the way the old sync notification code did.
@app.get("/api/dispatch/dead_letter")
async def get_dispatch_dead_letter(
    limit: int = Query(50, ge=1, le=DEAD_LETTER_MAX_ENTRIES),
    principal: Principal = Depends(get_current_principal),
):
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
            # A malformed entry should not poison the whole response.
            entries.append({"raw": str(raw), "parse_error": True})
    return {"count": len(entries), "entries": entries}

# Mount Frontend Dashboard Static Directory
frontend_dir = os.path.join(os.path.dirname(__file__), "../frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
