# 🏔️ VarunX: AI & IoT-Powered Early Warning System for Flash Floods & Landslides in Hilly Regions
### Smart India Hackathon 2026 Prototype · Problem Statement ID: SIH26192
**Organization:** Ministry of Home Affairs — NDRF, DM Division | **Theme:** Disaster Management

---

## 📌 Overview
**VarunX** is an AI and IoT-enabled early-warning software platform designed to mitigate flash flood and landslide disasters across vulnerable hilly catchments in India. It continuously ingests multi-source telemetry (rainfall intensity, flow/water level, slope movement, discharge) and predicts hyper-local village/ward-level risk levels (*Low, Medium, High, Critical*) alongside actionable evacuation lead times.

---

## ✅ Key Deliverables (Matching PRD 6.1 – 6.6)
- **Live Monitoring Dashboard**: Real-time sensor controls, rate of change indicators, risk scoring, and lead time estimation.
- **3D Valley & Slope Risk Model**: Interactive 3D Plotly valley terrain depicting river inundation and slope displacement zones.
- **Interactive GIS Catchment Map**: Geo-map plotting hilly catchments, live threat levels, and downstream ward vulnerabilities.
- **Historical Event Replay**: One-click replay of past historical flash flood and landslide events.
- **Vulnerable Zone Inventory**: Searchable inventory of vulnerable hilly zones with direct monitor hand-off.
- **Emergency Alert Gateway**: Multi-channel alert broadcast system (SMS, Email, Webhook, CAP XML for NDRF).

---

## 🚀 How to Run (3 Steps)

### 1. Install Dependencies
```bash
cd glof_ews
pip install -r requirements.txt
```

### 2. Launch Dashboard
```bash
streamlit run app.py
```
App opens in browser at `http://localhost:8501`.

---

## 📁 Project Architecture
```
glof_ews/
├── app.py                      # Main VarunX 5-tab Streamlit dashboard
├── mlTraining.txt              # Standalone external GPU model training guide
├── requirements.txt            # Project dependencies
├── schemas/
│   ├── __init__.py
│   └── sensor_data.py          # Pydantic schemas (SensorReading, AlertPayload)
├── utils/
│   ├── alert_dispatcher.py     # Multi-channel alert gateway (NDRF / SDMA)
│   ├── gis_mapper.py           # Plotly spatial GIS catchment mapper
│   ├── telemetry_stream.py     # Open-Meteo weather API & sensor stream tick generator
│   └── data_generator.py       # Synthetic training data generator
├── data/
│   ├── historical_glof_data.csv # Past event data
│   └── alerts_history.json     # Audit log for dispatched warnings
└── models/
    ├── glof_risk_model.pkl     # XGBoost Model
    ├── glof_risk_model_rf.pkl  # RandomForest Model
    ├── feature_cols.pkl
    └── metrics.json            # Model evaluation metrics
```

---

## 🧠 5-Level Architecture Progression Framework
1. **Level 1 (Static Geospatial Integration)**: Grounded on verified historical datasets (ISRO NRSC Landslide Atlas, GSI NLSM).
2. **Level 2 (Feature Engineering)**: Multi-dimensional environmental matrix fusing Bhuvan DEM elevation/slope with IMD gridded rainfall for Antecedent Moisture Condition (AMC).
3. **Level 3 (Supervised ML Architecture)**: XGBoost/RandomForest risk engine trained dynamically on learned environmental thresholds.
4. **Level 4 (Live Telemetry & API Fusion)**: Open-Meteo weather forecasts + MQTT cloud broker for physical IoT sensors.
5. **Level 5 (Dynamic Risk & CAP Warnings)**: Rate-of-change dynamic lead times + ITU CAP v1.2 XML alert dispatch for NDMA Sanchar Saathi broadcast.
