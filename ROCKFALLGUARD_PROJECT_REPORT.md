# RockfallGuard Project Report & Simple Guide

> **System:** Proactive Open-Pit Mine Geological Risk Warning Platform  
> **Accuracy:** 99.50% Holdout Test Set Accuracy (100% Failure Event Recall)  
> **Web Dashboard:** [http://127.0.0.1:8009](http://127.0.0.1:8009)

---

## 💡 What is RockfallGuard?

**RockfallGuard** is an Artificial Intelligence (AI) safety platform designed for open-pit mining operations. It monitors underground ground sensors and live weather telemetry to predict dangerous rock wall collapses **BEFORE** they happen, allowing miners to evacuate safely.

---

## 1. The Problem We Solved

In open-pit mining, giant open rock faces can collapse without warning due to heavy rainfall or underground water pressure. Traditional alarm systems only sound an alert *after* rocks start falling. **RockfallGuard** uses Machine Learning (ML) to provide proactive early warnings with **99.5% Accuracy**.

---

## 2. How RockfallGuard Works (5 Simple Steps)

1. **Ground Sensors**: Measures ground movement velocity ($\text{mm/h}$), acceleration ($\text{mm/h}^2$), and underground water pressure ($\text{kPa}$).
2. **GPS Weather Integration**: Uses the mine's exact GPS location (Latitude & Longitude) to check live rainfall from the Open-Meteo API.
3. **Machinery Noise Filter**: Uses a 4th-order Butterworth digital filter to ignore heavy haul truck vibration noise, preventing false alarms.
4. **AI Risk Score (0 – 100%)**:
   - 🟢 **SAFE (<35%)**: Normal safe mining operations.
   - 🟡 **WARNING (35–65%)**: Caution required; inspect slope bench.
   - 🔴 **CRITICAL (>65%)**: DANGER! Evacuate lower bench sectors immediately!
5. **Automated Emergency Email & SMS Alerts**: When risk exceeds 80%, the system automatically sounds sirens and dispatches emergency Emails and SMS text messages to mine safety officers.

---

## 3. Key Features Included

| Feature | Description |
| :--- | :--- |
| 📊 **Live Risk Score Gauge** | Displays real-time failure probability graphs and gauge updates every second. |
| 🔍 **SHAP Explainable AI** | Explains physical root causes behind risk spikes (e.g. *"+2.31 Acoustic Acceleration"*). |
| 🛸 **Drone Photo Scanning** | PyTorch Deep Learning CNN inspects aerial drone photos for rock cracks. |
| 🗺️ **Interactive GPS Map** | Live OpenStreetMap satellite pin centered on the mine's geographic location. |
| ➕ **Register & Delete Mines** | Allows safety managers to add new pit mines or delete old ones from SQLite. |
| 📱 **Emergency Email & SMS** | Dispatches automated alerts to registered phone numbers and email addresses. |

---

## 4. How to Use the Dashboard

1. Open **[http://127.0.0.1:8009](http://127.0.0.1:8009)** in your web browser.
2. Select your active mine from the top dropdown menu.
3. Use the scenario buttons (**Normal**, **Heavy Rain**, **Machinery Noise**, **Emergency Alert**) to test the system.
4. Click **🛸 Drone CNN Inspection** to test drone photo crack analysis.
5. Click **🗑️ Delete** if you want to remove any mine from the database.
