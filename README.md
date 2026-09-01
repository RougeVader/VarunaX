# 🏔️ VarunX (VarunaX) — Early Warning System (EWS)
### AI & IoT-Powered Flash Flood & Landslide Risk Prediction for Hilly Regions
**SIH Problem Statement:** SIH26192 • **Ministry:** Ministry of Home Affairs (NDRF, Disaster Management Division)

---

## 📌 Executive Summary
**VarunX** is an end-to-end mission-critical Early Warning System (EWS) designed to protect vulnerable Himalayan and mountainous communities from Glacial Lake Outburst Floods (GLOFs), flash floods, cloudbursts, and debris flow landslides.

The system ingests real-time IoT multi-modal telemetry (radar river stage, borehole inclinometer slope displacement, optical rain gauges, pore water pressure, soil moisture, turbidity), executes a 4-tier ML Risk Classifier (**Ensemble XGBoost + Random Forest** trained on CUDA GPU), computes dynamic evacuation lead times, renders an interactive **3D WebGL Valley Inundation Model**, and dispatches automated **ITU-T X.1303 Common Alerting Protocol (CAP v1.2 XML)** emergency broadcasts to NDRF, SDMA, and civil authorities.

---

## 🚀 Key Features

1. **🔴 Live Telemetry & Mission Control Engine (@st.fragment):**
   - Autonomous 10-second real-time streaming engine with persistent state.
   - Dual Mode: **Auto Simulation Mode** (dynamic stochastic hydrographs & rainstorms) and **Manual Scenario Override** (test extreme flood/landslide conditions live).
   - High-contrast tactical dark theme with CSS custom properties.

2. **🏔️ 3D Valley Terrain & Inundation Model (Three.js WebGL):**
   - High-fidelity 3D canyon topography with natural Himalayan elevation shading.
   - **Slope-Constrained Inundation Surface:** Water volume dynamically rises and expands along the physical contours of the mountain slopes without clipping.
   - **Raycasted Interactive 3D IoT Pins:** Hover over sensor nodes (S-01, S-07, Ward Hub, S-10) to view live telemetry HUD inspection cards.
   - **Persistent Viewport Memory:** Zoom, tilt, and pan angles are saved across 10-second refreshes.
   - **Active Landslide Slip Plane:** Visualizes glowing geotechnical failure scarps when slope displacement exceeds hazard thresholds.

3. **🗺️ Interactive GIS Spatial Map (Plotly Carto Dark):**
   - Visualizes critical catchment basins across Uttarakhand, Himachal Pradesh, Sikkim, and J&K (e.g., Kedarnath Valley / Mandakini Catchment).
   - Shows IoT sensor network coordinates, vulnerable downstream civil wards, high-risk moraine dams, and primary evacuation safe zones.

4. **⚡ 4-Tier ML Risk Classification & Lead Time Engine:**
   - Classifies risk into **Green (Normal)**, **Yellow (Advisory)**, **Orange (Warning)**, and **Red (Critical Emergency)**.
   - Calculates **Dynamic Evacuation Lead Time** using hydrodynamic catchment lag, Manning’s river channel velocity, and hydraulic distance.
   - Downstream Vulnerable Ward Impact & Response table with estimated time-to-impact and assigned evacuation centers.

5. **📢 ITU-T CAP v1.2 Emergency Alert Gateway:**
   - Formats and dispatches standard XML alerts compliant with ITU-T Recommendation X.1303 (compatible with NDMA Sachet, SDMA, and NDRF quick response).
   - Multi-channel dispatch: Emergency SMS, Broadcast Sirens, CAP XML, and WhatsApp Community Beacon.

6. **🔬 Offline Batch Diagnostic Hub & Analytics:**
   - Upload CSV/JSON field logs for high-throughput batch inference.
   - Correlation heatmaps, feature importance breakdown, and audit logs.

---

## System Architecture

```mermaid
flowchart TD
    %% Data Layer
    IoT["<b>IoT & Satellite Data Layer</b><br/><br/>Radar Gauges | Inclinometers | Rain Gauges | Piezometers"]
    
    %% Core Engine
    Core["<b>VarunX Real-Time Core Engine</b><br/><br/>• Multi-Sensor Fusion & Normalization (Pydantic)<br/>• 4-Tier ML Classifier (XGBoost GPU Hist + Random Forest)<br/>• Dynamic Hydrodynamic Evacuation Lead Time Model"]
    
    %% Outputs
    Dash["<b>Mission Control Dashboard</b><br/><br/>• Live 10s Telemetry Fragment<br/>• 3D WebGL Valley Inundation Model<br/>• Plotly Carto GIS Catchment Map<br/>• Downstream Ward Response Matrix"]
    
    Gate["<b>CAP Emergency Gateway</b><br/><br/>• ITU-T CAP v1.2 XML Feed<br/>• NDRF / SDMA Alert Dispatch<br/>• Multi-Channel Broadcast<br/>• Siren / SMS Triggers"]

    %% Routing
    IoT -- "Streaming Telemetry / REST API" --> Core
    
    Core --> Dash
    Core --> Gate

    %% Styling (Professional Minimalist Theme)
    style IoT fill:#f8f9fa,stroke:#adb5bd,stroke-width:1px,color:#212529
    style Core fill:#e9ecef,stroke:#6c757d,stroke-width:2px,color:#212529
    style Dash fill:#f8f9fa,stroke:#adb5bd,stroke-width:1px,color:#212529
    style Gate fill:#f8f9fa,stroke:#adb5bd,stroke-width:1px,color:#212529
    
    linkStyle default stroke:#6c757d,stroke-width:1.5px
---

### 🛰️ IoT & Satellite Data Layer
*Radar Gauges | Inclinometers | Rain Gauges | Piezometers*
↳ **Streaming Telemetry / REST API**

### ⚙️ VarunX Real-Time Core Engine
- **Multi-Sensor Fusion & Normalization** (`Pydantic SensorSchemas`)
- **4-Tier ML Classifier** (`XGBoost GPU Hist` + `Random Forest`)
- **Dynamic Hydrodynamic Evacuation Lead Time Model**

### 📤 Downstream Systems
| 🖥️ Mission Control Dashboard | 🚨 CAP Emergency Gateway |
| :--- | :--- |
| • Live 10s Telemetry Fragment | • ITU-T CAP v1.2 XML Feed |
| • 3D WebGL Valley Inundation Model | • NDRF / SDMA Alert Dispatch |
| • Plotly Carto GIS Catchment Map | • Multi-Channel Broadcast |
| • Downstream Ward Response Matrix | • Siren / SMS Triggers |

## 💻 How to Deploy on ANY Machine

### Prerequisites
- **Python:** Python 3.9, 3.10, 3.11, or 3.12 installed.
- **Git:** Installed on your system.
- **Hardware:** Runs on any standard CPU (Windows / Linux / macOS). *Optional: NVIDIA GPU with CUDA for accelerated model retraining.*

---

### Option 1: Quick Start (Windows)

1. **Clone the Repository:**
   `powershell
   git clone https://github.com/RougeVader/VarunaX.git
   cd VarunaX
   `

2. **Create and Activate a Virtual Environment:**
   `powershell
   python -m venv varunx_env
   .arunx_env\Scripts\Activate.ps1
   `
   *(If script execution is disabled in PowerShell, run: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass)*

3. **Install Dependencies:**
   `powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   `

4. **Launch the Application:**
   `powershell
   streamlit run app.py
   `
   The dashboard will automatically open in your default browser at http://localhost:8501.

---

### Option 2: Linux (Ubuntu / Debian / CentOS)

1. **Clone and Navigate:**
   `ash
   git clone https://github.com/RougeVader/VarunaX.git
   cd VarunaX
   `

2. **Set Up Python Virtual Environment:**
   `ash
   sudo apt-get update && sudo apt-get install -y python3 python3-pip python3-venv
   python3 -m venv varunx_env
   source varunx_env/bin/activate
   `

3. **Install Requirements & Run:**
   `ash
   pip install --upgrade pip
   pip install -r requirements.txt
   streamlit run app.py --server.port 8501 --server.address 0.0.0.0
   `

---

### Option 3: macOS (Apple Silicon M1/M2/M3 & Intel)

1. **Clone and Setup:**
   `ash
   git clone https://github.com/RougeVader/VarunaX.git
   cd VarunaX
   python3 -m venv varunx_env
   source varunx_env/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   `

2. **Run:**
   `ash
   streamlit run app.py
   `

---

### Option 4: Docker Deployment

1. **Build the Docker Image:**
   `ash
   docker build -t varunx-ews .
   `

2. **Run the Container:**
   `ash
   docker run -p 8501:8501 varunx-ews
   `
   Access at http://localhost:8501.

---

## 🧠 Machine Learning Model Training

The repository includes pre-trained production models in models/. To re-train the models with custom hyperparameter tuning or GPU acceleration:

`ash
python train_model.py
`

### Features used in ML Training (10 Core Parameters):
- 
ainfall_1h_mm & 
ainfall_24h_mm (Precipitation accumulation)
- low_water_level_m & discharge_m3s (Hydrodynamic stage & flow)
- slope_movement_mm & soil_moisture_pct (Geotechnical instability)
- pore_water_pressure_kpa & surface_temp_c (Subsurface saturation & snowmelt)
- 	urbidity_ntu & glacier_lake_area_change_pct (Sediment surge & moraine dam expansion)

---

## Repository Structure

For a clean, professional README, it is best practice to separate the visual directory tree from the detailed file descriptions. This prevents line-wrapping issues on smaller screens and keeps the documentation highly readable.

```text
VarunaX/
├── .streamlit/
│   └── config.toml
├── data/
│   ├── glacial_lakes_inventory/
│   ├── alerts_history.json
│   └── historical_varunx_data.csv
├── models/
│   ├── feature_cols.pkl
│   ├── metrics.json
│   ├── varunx_risk_model.pkl
│   └── varunx_risk_model_rf.pkl
├── schemas/
│   └── sensor_data.py
├── utils/
│   ├── alert_dispatcher.py
│   ├── data_generator.py
│   ├── gis_mapper.py
│   └── telemetry_stream.py
├── app.py
├── document.txt
├── README.md
├── requirements.txt
└── train_model.py
---

## 👥 Team & Acknowledgments
- **Project:** VarunX (VarunaX) Disaster Early Warning System
- **Hackathon:** Smart India Hackathon (SIH) 2026
- **Problem Code:** SIH26192
- **Target Organization:** Ministry of Home Affairs (MHA), National Disaster Response Force (NDRF), National Disaster Management Authority (NDMA).
