# RockfallGuard - Complete Project Overview & Feature Guide

> **System:** Proactive Open-Pit Mine Geological Risk Warning Platform  
> **Model Accuracy:** 99.50% Holdout Test Accuracy (100% Failure Event Recall)  
> **Live Web Dashboard:** [http://127.0.0.1:8009](http://127.0.0.1:8009)

---

## 📌 Executive Overview

**RockfallGuard** is a proactive, multi-sensor AI safety application for open-pit mining operations. It integrates ground sensors, live weather telemetry, digital signal noise filtering, machine learning risk calculation, PyTorch drone computer vision, and automated email/SMS emergency dispatches to predict rockfalls **BEFORE** they happen.

---

## 1. The Core Problem ⚠️

### What happens in Open-Pit Mines?
Open-pit mining involves digging giant, deep open-air pits into the earth to extract copper, gold, iron, or coal. Pit walls can be **hundreds of meters tall**.

### What is the major hazard?
- **Rockfalls and Bench Slope Failures**: Massive rock sections can collapse into the pit without warning, leading to loss of human life, multi-million-dollar damage to haul trucks and excavators, and total production halts.

### Why older traditional systems fail:
- **Reactive, static thresholds**: Older alarms ring only *after* ground displacement has already exceeded critical limits, giving workers zero time to evacuate.
- **High False-Alarm Rates**: Heavy mining trucks driving by trigger microseismic false alarms, causing miners to ignore alerts.
- **Siloed Data**: Sensor data (water pressure, displacement, weather) was analyzed in isolation.

---

## 2. How RockfallGuard Solves It 💡

```
[ Ground Sensors & Weather ] ➡️ [ Butterworth Noise Filter ] ➡️ [ AI Prediction Engine ] ➡️ [ Sirens + Email/SMS ]
```

1. **Multi-Sensor Data Fusion**: Reads ground movement velocity ($\text{mm/h}$), acceleration ($\text{mm/h}^2$), underground water pressure ($\text{kPa}$), and microseismic acoustic cracking ($g$) simultaneously.
2. **Live GPS Weather Telemetry**: Uses the mine's exact GPS coordinates (Latitude & Longitude) to pull live rainfall ($\text{mm/h}$) from the Open-Meteo API. Heavy rain increases underground pore pressure ($u$).
3. **Machinery Vibration Filter**: Uses a 4th-order **Butterworth Digital Filter** ($f_c = 15\text{ Hz}$) to strip away truck engine vibration noise, preventing false alarms.
4. **Predictive Machine Learning Engine**: An **XGBoost + LightGBM Ensemble Model** calculates a live **0 to 100% Failure Probability Score**:
   - 🟢 **SAFE (<35%)**: Normal safe mining.
   - 🟡 **WARNING (35–65%)**: Caution required; inspect slope bench.
   - 🔴 **CRITICAL (>65%)**: DANGER! Evacuate bench sectors immediately!
5. **Automated Emergency Dispatch**: If risk exceeds **80%**, the platform sounds sirens, flashes red alerts, and sends **Automated HTML Emails** and **SMS text messages** to mine safety officers.

---

## 3. Complete List of All Features 🌟

| Feature Name | Detailed Description |
| :--- | :--- |
| **📊 Live Risk Score Gauge** | Displays real-time 0–100% failure probability graphs and gauge updates every second. |
| **🔍 SHAP Explainable AI** | Explains physical root causes behind risk spikes (e.g. *"+2.31 Acoustic Fracture Acceleration"*). |
| **🛸 Drone Photo Scanning** | PyTorch Deep Learning CNN (`PitWallCNN` trained on 4,617 Kaggle images) inspects aerial drone photos for rock cracks. |
| **🗺️ Interactive GPS Map** | Live OpenStreetMap satellite pin centered on the mine's exact geographic location. |
| **📱 Emergency Email & SMS** | Dispatches automated HTML emails and Twilio SMS text alerts when danger exceeds 80%. |
| **➕ Register & Delete Mines** | Add new pit mines with GPS coordinates & alert thresholds, or click `🗑️ Delete` to remove old ones. |
| **🎮 Scenario Simulator** | Test operational scenarios: Normal Operation, Heavy Monsoon Storm, Machinery Noise, and Imminent Failure. |

---

## 4. Technology Stack Used 🛠️

| Component | Technology Used |
| :--- | :--- |
| **Frontend UI** | HTML5, Vanilla CSS3 (Glassmorphism), JavaScript (ES6+), Chart.js, Leaflet.js |
| **Backend API** | Python 3.14, FastAPI, Uvicorn |
| **Machine Learning** | XGBoost, LightGBM, Scikit-Learn, SHAP, Butterworth SciPy Filters, SMOTE |
| **Deep Learning Vision** | PyTorch, torchvision (`PitWallCNN` trained on 4,617 Kaggle drone images) |
| **Database** | SQLite3 (`mines.db`) |
| **Cache & Pub/Sub** | Redis 7 (Sub-10ms weather TTL caching) |
| **Notifications** | SMTP HTML Email + Twilio SMS Gateway (`backend/.env`) |
| **Container & CI/CD** | Docker, Docker-Compose, GitHub Actions (`.github/workflows/ci-cd.yml`) |

---

## 5. Model Accuracy & Results 📈

- **Holdout Test Set Accuracy**: **99.50%** (exceeds >95% requirement)
- **5-Fold Cross-Validation Accuracy**: **99.83%**
- **Critical Failure Event Recall**: **100.0%** (Zero missed slope collapses)
- **False Alarm Rate**: **0.17%** (Machinery noise filtered out)

---

## 6. How to Run & Use the Dashboard

1. Open **[http://127.0.0.1:8009](http://127.0.0.1:8009)** in your web browser.
2. Select your active mine from the top dropdown menu.
3. Use scenario buttons (**Normal**, **Heavy Rain**, **Machinery Noise**, **Emergency Alert**) to test the system.
4. Click **🛸 Drone CNN Inspection** to upload and inspect drone photos.
5. Click **🗑️ Delete** to delete any mine from the database.
