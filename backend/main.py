import os
import shutil
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from database import init_db, register_mine, get_all_mines, log_alert, get_recent_alerts
from weather_service import fetch_open_meteo_weather
from ml_engine import predict_rockfall_risk
from simulator import simulator_instance
from cv_engine import analyze_drone_pit_image

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

# Initialize Database on Startup
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

# --- API ENDPOINTS ---

@app.get("/api/health")
def health_check():
    return {"status": "online", "system": "RockfallGuard Proactive Slope Stability Engine"}

# 1. Mine Registration & Management
@app.post("/api/mines")
def register_new_mine(req: MineRegistrationRequest):
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

# 2. Real-Time Open-Meteo Weather Integration
@app.get("/api/weather/{lat}/{lon}")
def get_live_weather(lat: float, lon: float):
    return fetch_open_meteo_weather(lat, lon)

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
def get_mine_live_telemetry(mine_id: int, scenario: str = Query("normal")):
    mines = get_all_mines()
    target_mine = next((m for m in mines if m["id"] == mine_id), None)
    if not target_mine:
        target_mine = mines[0] if mines else {
            "id": 1, "name": "Default Open Pit Mine", "latitude": -4.05, "longitude": 137.11, "alert_threshold_pct": 70.0
        }
        
    # Fetch real live weather for mine GPS coordinates
    weather = fetch_open_meteo_weather(target_mine["latitude"], target_mine["longitude"])
    
    # Generate sensor telemetry frame
    telemetry = simulator_instance.generate_telemetry_frame(scenario=scenario, weather_data=weather)
    
    # Run ML Inference
    prediction_input = {**telemetry, "mine_id": mine_id}
    prediction = predict_rockfall_risk(prediction_input)
    
    # Auto log alert if risk exceeds mine threshold (e.g. >60-80%)
    alert_threshold = target_mine.get("alert_threshold_pct", 70.0)
    risk_pct = prediction["risk_percentage"]
    is_alert_triggered = risk_pct >= alert_threshold
    
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
        
    return {
        "mine": target_mine,
        "weather": weather,
        "telemetry": telemetry,
        "prediction": prediction,
        "alert_triggered": is_alert_triggered,
        "alert_threshold_pct": alert_threshold
    }

# 5. Alert History Endpoint
@app.get("/api/alerts")
def fetch_alerts():
    return get_recent_alerts(limit=50)

# 6. Upload Custom CSV Dataset for Retraining / Validation
@app.post("/api/upload_csv")
async def upload_custom_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
        
    upload_dir = os.path.join(os.path.dirname(__file__), "../data")
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, "custom_mine_upload.csv")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        df = pd.read_csv(file_path)
        rows, cols = df.shape
        return {
            "status": "success",
            "message": f"Successfully uploaded '{file.filename}' ({rows} rows, {cols} columns).",
            "columns": list(df.columns),
            "sample_head": df.head(3).to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file format: {str(e)}")

# 7. Drone UAV Bench Wall PyTorch CNN Image Analysis Endpoint
@app.api_route("/api/analyze_drone_image", methods=["GET", "POST"])
@app.api_route("/api/analyze_drone_image/", methods=["GET", "POST"])
async def analyze_drone_image_endpoint(file: Optional[UploadFile] = File(None)):
    if file is None:
        raise HTTPException(status_code=400, detail="Please select an image file to upload.")
    try:
        contents = await file.read()
        if not contents or len(contents) < 10:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        results = analyze_drone_pit_image(contents)
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing error: {str(e)}")

# Mount Frontend Dashboard Static Directory
frontend_dir = os.path.join(os.path.dirname(__file__), "../frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
