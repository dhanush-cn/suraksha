import sqlite3
import os
import json
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "mines.db")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Registered Open-Pit Mines Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS mines (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        company TEXT NOT NULL,
        location_name TEXT NOT NULL,
        latitude REAL NOT NULL,
        longitude REAL NOT NULL,
        pit_depth_m REAL DEFAULT 150.0,
        slope_angle_deg REAL DEFAULT 45.0,
        contact_email TEXT,
        contact_phone TEXT,
        alert_threshold_pct REAL DEFAULT 70.0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # 2. Historical Hazard Alerts Log Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS alert_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        mine_id INTEGER NOT NULL,
        risk_percentage REAL NOT NULL,
        risk_level TEXT NOT NULL,
        rainfall_mm REAL,
        pore_pressure_kpa REAL,
        velocity_mm_h REAL,
        seismic_rms_g REAL,
        top_shap_reason TEXT,
        triggered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (mine_id) REFERENCES mines(id)
    );
    """)
    
    # Insert initial default mining sites if table is empty
    cursor.execute("SELECT COUNT(*) FROM mines")
    if cursor.fetchone()[0] == 0:
        default_mines = [
            ("Grasberg Open-Pit Mine", "Freeport Copper-Gold", "Papua High Elevation Pit", -4.05, 137.11, 450.0, 48.0, "safety@grasbergmine.org", "+62 811-555-019"),
            ("Chuquicamata Mine", "Codelco Copper", "Atacama Pit Sector B", -22.31, -68.90, 850.0, 52.0, "geotech@codelco.cl", "+56 55-255-890"),
            ("Kalgoorlie Super Pit", "Northern Star Resources", "Goldfields Sector 4", -30.77, 121.50, 600.0, 44.0, "alerts@superpit.au", "+61 8-9022-1100")
        ]
        cursor.executemany("""
        INSERT INTO mines (name, company, location_name, latitude, longitude, pit_depth_m, slope_angle_deg, contact_email, contact_phone)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, default_mines)
        
    conn.commit()
    conn.close()
    print("[+] Database initialized successfully at:", DB_PATH)

def register_mine(name, company, location_name, latitude, longitude, pit_depth_m, slope_angle_deg, contact_email, contact_phone, alert_threshold_pct=70.0):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
        INSERT INTO mines (name, company, location_name, latitude, longitude, pit_depth_m, slope_angle_deg, contact_email, contact_phone, alert_threshold_pct)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (name, company, location_name, latitude, longitude, pit_depth_m, slope_angle_deg, contact_email, contact_phone, alert_threshold_pct))
        conn.commit()
        mine_id = cursor.lastrowid
        conn.close()
        return mine_id
    except sqlite3.IntegrityError:
        conn.close()
        raise ValueError(f"A mine named '{name}' is already registered.")

def get_all_mines():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM mines ORDER BY name ASC")
    mines = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return mines

def log_alert(mine_id, risk_percentage, risk_level, rainfall_mm, pore_pressure_kpa, velocity_mm_h, seismic_rms_g, top_shap_reason):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO alert_logs (mine_id, risk_percentage, risk_level, rainfall_mm, pore_pressure_kpa, velocity_mm_h, seismic_rms_g, top_shap_reason)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (mine_id, risk_percentage, risk_level, rainfall_mm, pore_pressure_kpa, velocity_mm_h, seismic_rms_g, top_shap_reason))
    conn.commit()
    conn.close()

def get_recent_alerts(limit=50):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT a.*, m.name as mine_name, m.company
    FROM alert_logs a
    JOIN mines m ON a.mine_id = m.id
    ORDER BY a.triggered_at DESC
    LIMIT ?
    """, (limit,))
    alerts = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return alerts

if __name__ == "__main__":
    init_db()
