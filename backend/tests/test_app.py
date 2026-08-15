import sys
import os
import pytest
from fastapi.testclient import TestClient

# Add backend directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../ml"))

from main import app
from ml_engine import predict_rockfall_risk, load_ml_artifacts

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "redis_connected" in data

def test_oauth2_jwt_login_admin():
    login_data = {"username": "admin", "password": "admin123"}
    res = client.post("/api/auth/login", json=login_data)
    assert res.status_code == 200
    data = res.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["user"]["role"] == "admin"

def test_oauth2_jwt_login_user():
    login_data = {"username": "grasberg_user", "password": "user123"}
    res = client.post("/api/auth/login", json=login_data)
    assert res.status_code == 200
    data = res.json()
    assert "access_token" in data
    assert data["user"]["role"] == "user"
    assert data["user"]["mine_id"] == 1

def test_user_tenant_data_isolation():
    # 1. Login as Grasberg User (Mine ID 1)
    login_res = client.post("/api/auth/login", json={"username": "grasberg_user", "password": "user123"})
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # 2. Access assigned Mine ID 1 -> Allowed (200 OK)
    res_own = client.get("/api/telemetry/1?scenario=normal", headers=headers)
    assert res_own.status_code == 200
    
    # 3. Access another Mine ID 2 -> Access Denied (403 Forbidden)
    res_other = client.get("/api/telemetry/2?scenario=normal", headers=headers)
    assert res_other.status_code == 403
    assert "Tenant Access Denied" in res_other.json()["detail"]

def test_admin_global_access():
    # 1. Login as Admin
    login_res = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # 2. Admin can access any mine telemetry
    res_mine1 = client.get("/api/telemetry/1?scenario=normal", headers=headers)
    assert res_mine1.status_code == 200
    
    res_mine2 = client.get("/api/telemetry/2?scenario=normal", headers=headers)
    assert res_mine2.status_code == 200

def test_user_forbidden_from_admin_actions():
    # 1. Login as Organization User
    login_res = client.post("/api/auth/login", json={"username": "grasberg_user", "password": "user123"})
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # 2. User cannot register new mine
    res_reg = client.post("/api/mines", json={"name": "Forbidden Mine", "company": "X", "location_name": "Y", "latitude": 0.0, "longitude": 0.0}, headers=headers)
    assert res_reg.status_code == 403
    
    # 3. User cannot delete a mine
    res_del = client.delete("/api/mines/1", headers=headers)
    assert res_del.status_code == 403

def test_ml_prediction_and_accuracy():
    sample_safe = {
        "rainfall_mm": 0.0,
        "humidity_pct": 50.0,
        "pore_pressure_kpa": 35.0,
        "displacement_mm": 5.0,
        "velocity_mm_h": 0.02,
        "acceleration_mm_h2": 0.0005,
        "raw_seismic_rms_g": 0.01
    }
    res_safe = predict_rockfall_risk(sample_safe)
    assert res_safe["risk_percentage"] < 35.0
    assert res_safe["risk_level"] == "Safe"
    
    sample_critical = {
        "rainfall_mm": 45.0,
        "humidity_pct": 95.0,
        "pore_pressure_kpa": 90.0,
        "displacement_mm": 80.0,
        "velocity_mm_h": 5.5,
        "acceleration_mm_h2": 1.2,
        "raw_seismic_rms_g": 0.45
    }
    res_crit = predict_rockfall_risk(sample_critical)
    assert res_crit["risk_percentage"] >= 65.0
    assert res_crit["risk_level"] == "Critical"
    assert len(res_crit["shap_explanations"]) > 0
