"""
VarunX: AI & IoT-Powered Early Warning System for Flash Floods & Landslides in Hilly Regions
SIH26192 · Smart India Hackathon 2026
Organization: Ministry of Home Affairs — NDRF, DM Division

Production UI (Streamlit + Plotly + High-Contrast Dark-Mode Design System)
Fixes:
- Removed pitch script expander from sidebar (clean command cockpit view)
- Smooth, flicker-free Plotly chart rendering using Streamlit @st.fragment
- Live Telemetry auto-refresh without full-page re-renders
"""

import os
import json
import time
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import joblib
from sklearn.ensemble import RandomForestClassifier

# Import local modules
from schemas.sensor_data import SensorReading, RiskPrediction, AlertPayload
from utils.alert_dispatcher import dispatch_emergency_alert, load_alert_history
from utils.gis_mapper import create_plotly_gis_map, CATCHMENT_ZONES_DATABASE
from utils.telemetry_stream import generate_telemetry_tick, fetch_live_weather
from utils.data_generator import generate_historical_data

# Streamlit Page Configuration
st.set_page_config(
    page_title="VarunX - Disaster Early Warning System (NDRF)",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# VarunX Dark Tactical Command Center Theme
def inject_custom_css():
    st.markdown("""
    <style>
        /* Base Canvas & Dark Theme Tokens */
        :root, .stApp {
            --vx-bg: #0E1117;
            --vx-card-bg: #161B26;
            --vx-border: #1E293B;
            --vx-border-subtle: #0F172A;
            --vx-text: #F8FAFC;
            --vx-text-muted: #94A3B8;
            --vx-accent: #F97316;
            --vx-accent-subtle: #38BDF8;
            --vx-sidebar-bg: #0B0F19;
            --vx-sidebar-border: #1E293B;
            --vx-shadow: rgba(0, 0, 0, 0.45);
            --vx-badge-bg: #1E293B;
            --vx-badge-color: #F97316;
            --vx-badge-border: #334155;
            --vx-table-th: #1E293B;
            --vx-table-hover: rgba(255, 255, 255, 0.04);
            --vx-pill-crit-bg: rgba(220, 38, 38, 0.25); --vx-pill-crit-text: #FCA5A5;
            --vx-pill-high-bg: rgba(234, 88, 12, 0.25); --vx-pill-high-text: #FDBA74;
            --vx-pill-med-bg: rgba(217, 119, 6, 0.25); --vx-pill-med-text: #FCD34D;
            --vx-pill-low-bg: rgba(22, 163, 74, 0.25); --vx-pill-low-text: #86EFAC;
        }

        .stApp {
            background-color: var(--vx-bg) !important;
            color: var(--vx-text) !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }
        
        .block-container {
            padding-top: 1.0rem !important;
            padding-bottom: 2rem !important;
            max-width: 98% !important;
        }
        
        /* Top Alert Banner */
        .alert-banner {
            padding: 12px 20px;
            border-radius: 10px;
            font-weight: 700;
            font-size: 0.98rem;
            letter-spacing: 0.5px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 15px;
            box-shadow: 0 4px 16px var(--vx-shadow);
        }
        .alert-banner-critical {
            background: linear-gradient(90deg, #DC2626 0%, #EA580C 70%, #C2410C 100%);
            color: #FFFFFF;
            border: 1px solid #B91C1C;
        }
        .alert-banner-high {
            background: linear-gradient(90deg, #EA580C 0%, #F97316 60%, #FB923C 100%);
            color: #FFFFFF;
            border: 1px solid #C2410C;
        }
        .alert-banner-medium {
            background: linear-gradient(90deg, #D97706 0%, #F59E0B 100%);
            color: #FFFFFF;
            border: 1px solid #B45309;
        }
        .alert-banner-low {
            background: linear-gradient(90deg, #059669 0%, #0D9488 100%);
            color: #FFFFFF;
            border: 1px solid #047857;
        }

        /* Unified Card Header */
        .card-header {
            font-size: 0.88rem;
            font-weight: 800;
            color: var(--vx-accent);
            letter-spacing: 0.6px;
            text-transform: uppercase;
            margin-bottom: 0.6rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        
        /* Metric Cards */
        div[data-testid="stMetric"] {
            background-color: var(--vx-card-bg) !important;
            border: 1px solid var(--vx-border) !important;
            padding: 0.8rem 1rem !important;
            border-radius: 12px !important;
            box-shadow: 0 2px 10px var(--vx-shadow) !important;
        }
        div[data-testid="stMetricLabel"] label, div[data-testid="stMetricLabel"] p {
            color: var(--vx-text-muted) !important;
            font-size: 0.82rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.4px !important;
            text-transform: uppercase;
        }
        div[data-testid="stMetricValue"] div, div[data-testid="stMetricValue"] p {
            color: var(--vx-accent) !important;
            font-size: 1.65rem !important;
            font-weight: 800 !important;
        }
        div[data-testid="stMetricDelta"] div, div[data-testid="stMetricDelta"] p {
            color: var(--vx-accent-subtle) !important;
            font-weight: 700 !important;
        }
        
        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: var(--vx-sidebar-bg) !important;
            border-right: 1px solid var(--vx-sidebar-border) !important;
        }
        section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] label {
            color: var(--vx-text) !important;
            font-weight: 600;
        }
        
        /* Plotly Chart Container */
        .stPlotlyChart {
            border-radius: 12px;
            border: 1px solid var(--vx-border);
            background-color: var(--vx-card-bg);
            padding: 2px;
            box-shadow: 0 2px 10px var(--vx-shadow);
        }

        /* Live Pulse Indicator */
        .live-pulse-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: var(--vx-accent);
            display: inline-block;
            box-shadow: 0 0 0 rgba(234, 88, 12, 0.6);
            animation: pulse-live 1.6s infinite;
        }
        .live-pulse-dot.offline {
            background: var(--vx-text-muted);
            animation: none;
        }
        @keyframes pulse-live {
            0% { box-shadow: 0 0 0 0 rgba(234, 88, 12, 0.55); }
            70% { box-shadow: 0 0 0 9px rgba(234, 88, 12, 0); }
            100% { box-shadow: 0 0 0 0 rgba(234, 88, 12, 0); }
        }

        /* Custom Risk Pills */
        .risk-pill {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 999px;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.3px;
            text-transform: uppercase;
        }
        .pill-critical { background: var(--vx-pill-crit-bg); color: var(--vx-pill-crit-text); border: 1px solid #F87171; }
        .pill-high { background: var(--vx-pill-high-bg); color: var(--vx-pill-high-text); border: 1px solid #FB923C; }
        .pill-medium { background: var(--vx-pill-med-bg); color: var(--vx-pill-med-text); border: 1px solid #FBBF24; }
        .pill-low { background: var(--vx-pill-low-bg); color: var(--vx-pill-low-text); border: 1px solid #4ADE80; }

        /* Custom Table Styling */
        .varunx-table-wrap {
            background: var(--vx-card-bg);
            border: 1px solid var(--vx-border);
            border-radius: 12px;
            overflow: hidden;
            margin-top: 0.8rem;
            box-shadow: 0 2px 10px var(--vx-shadow);
        }
        .varunx-table {
            width: 100%;
            border-collapse: collapse;
            color: var(--vx-text);
            font-size: 0.86rem;
        }
        .varunx-table th {
            background: var(--vx-table-th);
            padding: 10px 14px;
            text-align: left;
            color: var(--vx-accent);
            font-weight: 700;
            font-size: 0.78rem;
            text-transform: uppercase;
            border-bottom: 1px solid var(--vx-border);
        }
        .varunx-table td {
            padding: 11px 14px;
            border-bottom: 1px solid var(--vx-border);
        }
        .varunx-table tr:hover {
            background: var(--vx-table-hover);
        }

        /* SMS Preview */
        .sms-bubble {
            background: var(--vx-card-bg);
            border: 1px solid var(--vx-border);
            border-radius: 16px 16px 16px 4px;
            padding: 0.9rem 1.1rem;
            color: var(--vx-text);
            font-size: 0.88rem;
            line-height: 1.45;
            max-width: 440px;
            box-shadow: 0 4px 14px var(--vx-shadow);
            margin-top: 0.5rem;
        }
        .sms-meta {
            font-size: 0.72rem;
            color: var(--vx-text-muted);
            margin-top: 0.3rem;
        }
    </style>
    """, unsafe_allow_html=True)


VARUNX_FEATURE_COLS = [
    "rainfall_1h_mm", "rainfall_3h_mm", "rainfall_24h_mm",
    "flow_water_level_m", "water_level_rate_m_h",
    "slope_movement_mm", "tilt_rate_mm_h",
    "discharge_m3s", "air_temp_c", "surface_temp_c"
]


@st.cache_resource
def load_or_train_model():
    model_path = "models/varunx_risk_model_rf.pkl"
    features_path = "models/feature_cols.pkl"
    
    if os.path.exists(model_path) and os.path.exists(features_path):
        try:
            model = joblib.load(model_path)
            features = joblib.load(features_path)
            if len(features) == len(VARUNX_FEATURE_COLS):
                return model, features
        except Exception:
            pass
            
    df = generate_historical_data(1500)
    X = df[VARUNX_FEATURE_COLS]
    y = df["risk_level"]
    rf = RandomForestClassifier(n_estimators=80, max_depth=6, random_state=42)
    rf.fit(X, y)
    return rf, VARUNX_FEATURE_COLS


@st.cache_data
def load_historical_events():
    path = "data/historical_varunx_data.csv"
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


def get_risk_label(level: int) -> str:
    labels = {0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "CRITICAL"}
    return labels.get(level, "UNKNOWN")


def estimate_lead_time(level: int, flow_rate: float, slope_movement: float) -> float:
    base = 36.0 if level == 0 else (14.0 if level == 1 else (6.0 if level == 2 else 2.5))
    if flow_rate > 0.5 or slope_movement > 35.0:
        base *= 0.6
    return max(0.5, round(base, 1))


def format_hms(total_seconds: float) -> str:
    total_seconds = max(0, int(total_seconds))
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def build_cap_xml_preview(zone: str, risk_label: str, lead_time_hours: float, trigger_reason: str, alert_id: str) -> str:
    sent_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S+05:30")
    severity_map = {"LOW": "Minor", "MEDIUM": "Moderate", "HIGH": "Severe", "CRITICAL": "Extreme"}
    urgency_map = {"LOW": "Future", "MEDIUM": "Expected", "HIGH": "Immediate", "CRITICAL": "Immediate"}
    return f"""<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
  <identifier>{alert_id}</identifier>
  <sender>ndrf.varunx@dm-division.gov.in</sender>
  <sent>{sent_time}</sent>
  <status>Actual</status>
  <msgType>Alert</msgType>
  <scope>Public</scope>
  <info>
    <event>Flash Flood / Landslide Early Warning</event>
    <urgency>{urgency_map.get(risk_label, "Expected")}</urgency>
    <severity>{severity_map.get(risk_label, "Moderate")}</severity>
    <certainty>Likely</certainty>
    <areaDesc>{zone}</areaDesc>
    <effective>{sent_time}</effective>
    <expires>+{lead_time_hours:.1f}h</expires>
    <headline>{risk_label} RISK — {zone}</headline>
    <description>{trigger_reason}</description>
  </info>
</alert>"""


def render_sms_preview(zone: str, risk_label: str, lead_time_hours: float):
    body = (
        f"⚠️ VarunX NDRF ALERT: {risk_label} risk detected in {zone}. "
        f"Estimated evacuation window: {lead_time_hours:.1f} hrs. "
        f"Move to designated safe zones immediately. Follow local DM authority instructions."
    )
    st.markdown(
        f"""<div class="sms-bubble">{body}<div class="sms-meta">NDRF-ALERT · delivered via SMS gateway · just now</div></div>""",
        unsafe_allow_html=True
    )


def render_3d_valley_component(flow_water_level: float, slope_movement: float, selected_zone: str):
    html_code = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; padding: 0; overflow: hidden; background: #161B26;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }}
    #canvas-container {{
      width: 100%; height: 440px; position: relative; border-radius: 10px; overflow: hidden;
    }}
    .hud-overlay {{
      position: absolute; top: 10px; left: 12px; pointer-events: none;
      background: rgba(14, 17, 23, 0.85); backdrop-filter: blur(6px);
      border: 1px solid #1E293B; border-radius: 8px; padding: 6px 12px;
      color: #F8FAFC; font-size: 11px; z-index: 10;
    }}
    .hud-stat {{ color: #F97316; font-weight: 700; }}
    .hint-overlay {{
      position: absolute; bottom: 8px; right: 12px; pointer-events: none;
      color: #64748B; font-size: 10px; z-index: 10;
      background: rgba(14, 17, 23, 0.6); padding: 3px 8px; border-radius: 4px;
    }}
    #pin-tooltip {{
      position: absolute; display: none; pointer-events: none; z-index: 50;
      background: rgba(15, 23, 42, 0.95); border: 1px solid #F97316;
      border-radius: 8px; padding: 8px 12px; box-shadow: 0 4px 18px rgba(0,0,0,0.6);
      backdrop-filter: blur(8px); min-width: 180px;
    }}
  </style>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
  <div id="canvas-container">
    <div class="hud-overlay">
      <div>🏔️ <b>{selected_zone}</b></div>
      <div style="margin-top:2px;">🌊 River Stage: <span class="hud-stat">{flow_water_level:.1f} m</span> &nbsp;|&nbsp; ⛰️ Slope: <span class="hud-stat">{slope_movement:.1f} mm</span></div>
    </div>
    <div class="hint-overlay">🖱️ Hover on Pins to Inspect · Drag to Rotate · Scroll to Zoom</div>
    <div id="pin-tooltip"></div>
  </div>
  <script>
    (function() {{
      var targetWaterDepth = {flow_water_level};
      var targetSlopeMovement = {slope_movement};
      var container = document.getElementById('canvas-container');
      var tooltipEl = document.getElementById('pin-tooltip');
      var width = container.clientWidth || window.innerWidth;
      var height = 440;

      // 1. Scene & Camera Setup
      var scene = new THREE.Scene();
      scene.background = new THREE.Color(0x161B26);
      scene.fog = new THREE.FogExp2(0x161B26, 0.032);

      var camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
      var defaultPos = {{ x: 9.5, y: 7.2, z: 10.5 }};
      var defaultTarget = {{ x: 0, y: -0.5, z: 0 }};

      // Persistent Camera Position Lock across 10s refreshes
      try {{
        if (window.parent && window.parent._varunx_3d_cam) {{
          var c = window.parent._varunx_3d_cam;
          camera.position.set(c.x, c.y, c.z);
          defaultTarget = {{ x: c.tx, y: c.ty, z: c.tz }};
        }} else {{
          camera.position.set(defaultPos.x, defaultPos.y, defaultPos.z);
        }}
      }} catch(e) {{
        camera.position.set(defaultPos.x, defaultPos.y, defaultPos.z);
      }}

      var renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true, powerPreference: "high-performance" }});
      renderer.setSize(width, height);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.shadowMap.enabled = true;
      container.appendChild(renderer.domElement);

      var controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.08;
      controls.target.set(defaultTarget.x, defaultTarget.y, defaultTarget.z);
      controls.maxPolarAngle = Math.PI / 2 - 0.05;
      controls.minDistance = 3;
      controls.maxDistance = 28;

      controls.addEventListener('change', function() {{
        try {{
          if (window.parent) {{
            window.parent._varunx_3d_cam = {{
              x: camera.position.x, y: camera.position.y, z: camera.position.z,
              tx: controls.target.x, ty: controls.target.y, tz: controls.target.z
            }};
          }}
        }} catch(e) {{}}
      }});

      // 2. Tactical Lighting
      var hemiLight = new THREE.HemisphereLight(0xffffff, 0x161B26, 0.85);
      scene.add(hemiLight);

      var dirLight = new THREE.DirectionalLight(0xffeedd, 1.2);
      dirLight.position.set(12, 18, 10);
      dirLight.castShadow = true;
      scene.add(dirLight);

      var ambLight = new THREE.AmbientLight(0x384B66, 0.4);
      scene.add(ambLight);

      // 3. Valley Topography Mesh
      function getTerrainElevation(x, z) {{
        var riverPath = 0.45 * Math.sin(0.7 * z);
        var distToRiver = Math.abs(x - riverPath);
        var y = 0.14 * (x*x) + 0.03 * (z*z) + 0.25 * Math.cos(0.8*x) * Math.cos(0.5*z) - 2.8;
        y -= 0.6 * Math.exp(-(distToRiver*distToRiver) / 1.5);
        return y;
      }}

      var gridSize = 64;
      var terrainGeo = new THREE.PlaneGeometry(14, 14, gridSize, gridSize);
      terrainGeo.rotateX(-Math.PI / 2);

      var pos = terrainGeo.attributes.position;
      var colors = [];
      var color = new THREE.Color();

      for (var i = 0; i < pos.count; i++) {{
        var x = pos.getX(i);
        var z = pos.getZ(i);
        var y = getTerrainElevation(x, z);
        pos.setY(i, y);

        // Natural Himalayan Topographic Gradient
        if (y < -2.4) {{
          color.setRGB(0.16, 0.32, 0.18); // River gorge vegetation
        }} else if (y < -0.8) {{
          color.setRGB(0.45, 0.40, 0.32); // Rock talus slopes
        }} else if (y < 1.2) {{
          color.setRGB(0.68, 0.62, 0.55); // High canyon crags
        }} else {{
          color.setRGB(0.88, 0.90, 0.94); // Snow/granite peaks
        }}
        colors.push(color.r, color.g, color.b);
      }}
      terrainGeo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
      terrainGeo.computeVertexNormals();

      var terrainMat = new THREE.MeshStandardMaterial({{
        vertexColors: true,
        roughness: 0.85,
        metalness: 0.1
      }});
      var terrainMesh = new THREE.Mesh(terrainGeo, terrainMat);
      terrainMesh.receiveShadow = true;
      scene.add(terrainMesh);

      // 4. Slope-Constrained Inundation Water Mesh (Topologically Bounded by Valley Walls)
      var waterGridSize = 64;
      var waterGeo = new THREE.PlaneGeometry(14, 14, waterGridSize, waterGridSize);
      waterGeo.rotateX(-Math.PI / 2);

      var targetWaterElevation = -3.2 + (targetWaterDepth / 20.0) * 4.8;
      var currentWaterElevation = targetWaterElevation;
      var wPos = waterGeo.attributes.position;
      var waterColors = [];
      var wCol = new THREE.Color();
      var isCriticalFlood = targetWaterDepth > 10.0;
      var waterBaseHex = isCriticalFlood ? 0xDC2626 : 0x0284C7;

      for (var i = 0; i < wPos.count; i++) {{
        var xVal = wPos.getX(i);
        var zVal = wPos.getZ(i);
        var tElev = getTerrainElevation(xVal, zVal);

        if (tElev < targetWaterElevation) {{
          // Inundated basin
          wPos.setY(i, targetWaterElevation);
          wCol.setHex(waterBaseHex);
        }} else {{
          // Dry valley slope above waterline - clamped just below terrain
          wPos.setY(i, tElev - 0.05);
          wCol.setRGB(0.09, 0.11, 0.15); // blends seamlessly with dark background
        }}
        waterColors.push(wCol.r, wCol.g, wCol.b);
      }}
      waterGeo.setAttribute('color', new THREE.Float32BufferAttribute(waterColors, 3));
      waterGeo.computeVertexNormals();

      var waterMat = new THREE.MeshStandardMaterial({{
        vertexColors: true,
        transparent: true,
        opacity: 0.88,
        roughness: 0.15,
        metalness: 0.7
      }});
      var waterMesh = new THREE.Mesh(waterGeo, waterMat);
      scene.add(waterMesh);

      // 5. Landslide Failure Scarp
      var scarpGeo = new THREE.PlaneGeometry(3.2, 3.2, 16, 16);
      scarpGeo.rotateX(-Math.PI / 2);
      var scarpMat = new THREE.MeshStandardMaterial({{
        color: 0xEF4444,
        roughness: 0.7,
        emissive: 0x7F1D1D,
        emissiveIntensity: 0.45
      }});
      var scarpMesh = new THREE.Mesh(scarpGeo, scarpMat);
      scarpMesh.position.set(-3.2, -0.6, 0.1);
      scarpMesh.rotation.z = 0.25;
      scarpMesh.visible = (targetSlopeMovement > 15.0);
      scene.add(scarpMesh);

      // 6. Interactive 3D IoT Sensor Pins with Raycasting
      var interactableObjects = [];

      function createSensorBeacon(x, y, z, colorHex, info) {{
        var group = new THREE.Group();
        group.position.set(x, y, z);

        var stemGeo = new THREE.CylinderGeometry(0.04, 0.04, 1.2, 8);
        var stemMat = new THREE.MeshBasicMaterial({{ color: 0x94A3B8 }});
        var stem = new THREE.Mesh(stemGeo, stemMat);
        stem.position.y = 0.6;
        group.add(stem);

        var beaconGeo = new THREE.SphereGeometry(0.26, 16, 16);
        var beaconMat = new THREE.MeshStandardMaterial({{
          color: colorHex,
          emissive: colorHex,
          emissiveIntensity: 0.65
        }});
        var beacon = new THREE.Mesh(beaconGeo, beaconMat);
        beacon.position.y = 1.2;
        beacon.userData = info;
        beacon.userData.colorHex = '#' + colorHex.toString(16).padStart(6, '0');
        group.add(beacon);
        interactableObjects.push(beacon);

        // Pulsing Ring
        var ringGeo = new THREE.RingGeometry(0.32, 0.44, 16);
        ringGeo.rotateX(-Math.PI / 2);
        var ringMat = new THREE.MeshBasicMaterial({{ color: colorHex, transparent: true, opacity: 0.7, side: THREE.DoubleSide }});
        var ring = new THREE.Mesh(ringGeo, ringMat);
        ring.position.y = 1.2;
        group.add(ring);

        scene.add(group);
        return {{ group: group, beacon: beacon, ring: ring }};
      }}

      var b1 = createSensorBeacon(0.1, -1.8, -2.0, 0x0284C7, {{
        name: "S-01: Radar River Gauge",
        valKey: "Current River Stage",
        val: "{flow_water_level:.1f} m",
        status: "Active Real-Time Telemetry Stream"
      }});

      var b2 = createSensorBeacon(-3.4, -0.2, 0.2, (targetSlopeMovement > 15 ? 0xDC2626 : 0xF97316), {{
        name: "S-07: Borehole Inclinometer",
        valKey: "Slope Displacement",
        val: "{slope_movement:.1f} mm",
        status: (targetSlopeMovement > 15 ? "CRITICAL HAZARD - Active Slope Failure" : "NORMAL Baseline")
      }});

      var b3 = createSensorBeacon(0.4, 0.8, 4.2, 0x16A34A, {{
        name: "Downstream Ward Hub",
        valKey: "Civil Settlement Zone",
        val: "Primary Evacuation Hub",
        status: "NDRF Quick Response Reach"
      }});

      var b4 = createSensorBeacon(3.2, 0.2, -3.0, 0xF97316, {{
        name: "S-10: Optical Rain Gauge",
        valKey: "Precipitation Monitoring",
        val: "Optical Disdrometer Stream",
        status: "Continuous 24h Catchment Ingestion"
      }});

      // 7. Interactive Hover Tooltip via Raycaster
      var raycaster = new THREE.Raycaster();
      var mouse = new THREE.Vector2();

      function onPointerMove(e) {{
        var rect = renderer.domElement.getBoundingClientRect();
        mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

        raycaster.setFromCamera(mouse, camera);
        var intersects = raycaster.intersectObjects(interactableObjects);

        if (intersects.length > 0) {{
          renderer.domElement.style.cursor = 'pointer';
          var hitObj = intersects[0].object;
          var data = hitObj.userData;
          tooltipEl.innerHTML = 
            '<div style="font-weight:800;color:' + data.colorHex + ';font-size:12px;margin-bottom:3px;">' + data.name + '</div>' +
            '<div style="color:#F8FAFC;font-size:11px;"><b>' + data.valKey + ':</b> <span style="color:#F97316;font-weight:700;">' + data.val + '</span></div>' +
            '<div style="color:#94A3B8;font-size:10px;margin-top:3px;"><b>Status:</b> ' + data.status + '</div>';
          tooltipEl.style.display = 'block';
          tooltipEl.style.left = (e.clientX - rect.left + 14) + 'px';
          tooltipEl.style.top = (e.clientY - rect.top - 20) + 'px';
        }} else {{
          renderer.domElement.style.cursor = 'default';
          tooltipEl.style.display = 'none';
        }}
      }}
      renderer.domElement.addEventListener('mousemove', onPointerMove);

      // 8. Render Loop with Contour Flood Inundation & Wave Dynamics
      var clock = new THREE.Clock();
      function animate() {{
        requestAnimationFrame(animate);
        var elapsed = clock.getElapsedTime();

        // Smoothly lerp water level to target
        currentWaterElevation += (targetWaterElevation - currentWaterElevation) * 0.08;

        var curPos = waterGeo.attributes.position;
        var curCol = waterGeo.attributes.color;

        for (var i = 0; i < curPos.count; i++) {{
          var x = curPos.getX(i);
          var z = curPos.getZ(i);
          var tE = getTerrainElevation(x, z);

          if (tE < currentWaterElevation) {{
            // Water rises and covers canyon bed up to valley slope wall
            var wave = 0.035 * Math.sin(x * 3.0 + elapsed * 2.5) + 0.025 * Math.cos(z * 2.2 + elapsed * 2.0);
            curPos.setY(i, currentWaterElevation + wave);
            curCol.setXYZ(i, isCriticalFlood ? 0.86 : 0.01, isCriticalFlood ? 0.15 : 0.52, isCriticalFlood ? 0.15 : 0.78);
          }} else {{
            // Dry mountain slope above waterline
            curPos.setY(i, tE - 0.05);
            curCol.setXYZ(i, 0.09, 0.11, 0.15);
          }}
        }}
        waterGeo.computeVertexNormals();
        curPos.needsUpdate = true;
        curCol.needsUpdate = true;

        // Pulse beacon rings
        var scale = 1.0 + 0.22 * Math.sin(elapsed * 4.0);
        b1.ring.scale.set(scale, scale, scale);
        b2.ring.scale.set(scale, scale, scale);
        b3.ring.scale.set(scale, scale, scale);
        b4.ring.scale.set(scale, scale, scale);

        controls.update();
        renderer.render(scene, camera);
      }}
      animate();

      window.addEventListener('resize', function() {{
        var newW = container.clientWidth || window.innerWidth;
        camera.aspect = newW / height;
        camera.updateProjectionMatrix();
        renderer.setSize(newW, height);
      }});
    }})();
  </script>
</body>
</html>"""
    st.components.v1.html(html_code, height=445)


def create_3d_valley_terrain(flow_water_level: float, slope_movement: float, selected_zone: str):
    # Plotly fallback compatibility
    x = np.linspace(-6, 6, 40)
    y = np.linspace(-6, 6, 40)
    X, Y = np.meshgrid(x, y)
    river_path = 0.45 * np.sin(0.7 * Y)
    dist_to_river = np.abs(X - river_path)
    Z_valley = 0.14 * (X**2) + 0.03 * (Y**2) - 0.6 * np.exp(-(dist_to_river**2) / 1.5) - 2.8
    fig = go.Figure(go.Surface(x=X, y=Y, z=Z_valley, showscale=False, opacity=0.95))
    fig.update_layout(height=440, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)", uirevision="varunx_3d_lock")
    return fig


def create_catchment_geomap(selected_zone: str, risk_level: int, flow_water_level: float, slope_movement: float):
    z_info = CATCHMENT_ZONES_DATABASE.get(selected_zone, CATCHMENT_ZONES_DATABASE["Kedarnath Valley (Mandakini Catchment)"])
    center_lat = z_info["lat"]
    center_lon = z_info["lon"]
    
    # Generate sensor node network around active catchment
    nodes = [
        {"id": "S-01", "name": "Radar River Gauge", "lat": center_lat + 0.005, "lon": center_lon - 0.004, "type": "Hydro", "color": "#0284C7", "size": 13},
        {"id": "S-02", "name": "Acoustic Flow Sensor", "lat": center_lat - 0.008, "lon": center_lon + 0.003, "type": "Discharge", "color": "#0284C7", "size": 11},
        {"id": "S-07", "name": "Borehole Inclinometer", "lat": center_lat + 0.012, "lon": center_lon + 0.006, "type": "Slope Failure", "color": "#DC2626" if slope_movement > 15 else "#EA580C", "size": 14},
        {"id": "S-09", "name": "Geotechnical Extensometer", "lat": center_lat + 0.015, "lon": center_lon + 0.002, "type": "Slope", "color": "#EA580C", "size": 11},
        {"id": "S-10", "name": "Optical Rain Gauge", "lat": center_lat - 0.012, "lon": center_lon - 0.008, "type": "Meteorological", "color": "#F97316", "size": 11}
    ]
    
    # Add downstream vulnerable wards
    for idx, w in enumerate(z_info["vulnerable_wards"]):
        w_color = "#DC2626" if risk_level == 3 else ("#EA580C" if risk_level == 2 else ("#D97706" if risk_level == 1 else "#16A34A"))
        nodes.append({
            "id": f"Ward-{idx+1}",
            "name": w["name"],
            "lat": w["lat"],
            "lon": w["lon"],
            "type": f"Population: {w['population']:,} · Dist: {w['distance_km']}km",
            "color": w_color,
            "size": 16
        })
        
    df_nodes = pd.DataFrame(nodes)
    
    if hasattr(go, "Scattermap"):
        fig = go.Figure(go.Scattermap(
            lat=df_nodes["lat"],
            lon=df_nodes["lon"],
            mode="markers+text",
            marker=dict(size=df_nodes["size"], color=df_nodes["color"], opacity=0.95),
            text=df_nodes["id"],
            textposition="top right",
            textfont=dict(size=9, color="#94A3B8"),
            hovertemplate="<b>%{customdata[0]} (%{text})</b><br>%{customdata[1]}<extra></extra>",
            customdata=df_nodes[["name", "type"]].values
        ))
        fig.update_layout(
            map_style="carto-darkmatter",
            map=dict(center=dict(lat=center_lat, lon=center_lon), zoom=11.6),
            margin=dict(l=0, r=0, b=0, t=10),
            height=440,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            uirevision="varunx_map_lock"
        )
    elif hasattr(go, "Scattermapbox"):
        fig = go.Figure(go.Scattermapbox(
            lat=df_nodes["lat"],
            lon=df_nodes["lon"],
            mode="markers+text",
            marker=dict(size=df_nodes["size"], color=df_nodes["color"], opacity=0.95),
            text=df_nodes["id"],
            textposition="top right",
            textfont=dict(size=9, color="#94A3B8"),
            hovertemplate="<b>%{customdata[0]} (%{text})</b><br>%{customdata[1]}<extra></extra>",
            customdata=df_nodes[["name", "type"]].values
        ))
        fig.update_layout(
            mapbox_style="carto-darkmatter",
            mapbox=dict(center=dict(lat=center_lat, lon=center_lon), zoom=11.6),
            margin=dict(l=0, r=0, b=0, t=10),
            height=440,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            uirevision="varunx_map_lock"
        )
    else:
        fig = px.scatter_geo(
            df_nodes, lat="lat", lon="lon", text="id", color="color", size="size"
        )
        fig.update_layout(height=440, margin=dict(l=0, r=0, b=0, t=10), uirevision="varunx_map_lock")
        
    return fig


def create_rainfall_river_stage_trend(hist_df: pd.DataFrame, risk_level: int, current_flow: float):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    if len(hist_df) > 1:
        x_vals = hist_df["timestamp"]
        # Generate correlated rainfall series
        rain_vals = [max(2.0, round(f * 3.8 + np.random.uniform(-1.5, 1.5), 1)) for f in hist_df["flow_water_level_m"]]
        flow_vals = hist_df["flow_water_level_m"]
    else:
        now = datetime.now()
        x_vals = [(now - pd.Timedelta(hours=24-i)).strftime("%H:00") for i in range(24)]
        rain_vals = np.random.uniform(10, 118, 24)
        flow_vals = np.linspace(2.5, current_flow, 24)
    
    # 1. 24h Rainfall Trace (Primary Y, Warm Orange Area Fill)
    fig.add_trace(
        go.Scatter(
            x=x_vals, y=rain_vals,
            name="Rainfall (mm)",
            line=dict(color="#F97316", width=2),
            fill="tozeroy",
            fillcolor="rgba(249, 115, 22, 0.18)"
        ),
        secondary_y=False
    )
    
    # 2. River Stage Flow Trace (Secondary Y, Sky Blue Line)
    fig.add_trace(
        go.Scatter(
            x=x_vals, y=flow_vals,
            name="River Stage (m)",
            line=dict(color="#0284C7", width=2.8)
        ),
        secondary_y=True
    )
    
    # Critical River Flood Threshold Line
    fig.add_hline(
        y=10.0, line_dash="dash", line_color="#DC2626", line_width=1.5,
        annotation_text="Critical River Stage (10m)", annotation_position="top left",
        annotation_font=dict(size=10, color="#DC2626"),
        secondary_y=True
    )
    
    fig.update_layout(
        height=260, margin=dict(l=5, r=5, t=10, b=5),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94A3B8", size=10),
        legend=dict(orientation="h", y=1.2, x=0.0, font=dict(color="#94A3B8")),
        xaxis=dict(showgrid=True, gridcolor="rgba(148, 163, 184, 0.2)"),
        yaxis=dict(title="Rainfall (mm)", gridcolor="rgba(148, 163, 184, 0.2)", color="#EA580C"),
        yaxis2=dict(title="River Stage (m)", showgrid=False, color="#0284C7"),
        uirevision="varunx_trend_rain"
    )
    return fig


def create_slope_displacement_trend(hist_df: pd.DataFrame, current_slope: float):
    fig = go.Figure()
    
    if len(hist_df) > 1:
        x_vals = hist_df["timestamp"]
        s_vals = hist_df["slope_movement_mm"]
        borehole_strain = [round(s * 0.72 + np.random.uniform(-0.4, 0.4), 2) for s in s_vals]
    else:
        now = datetime.now()
        x_vals = [(now - pd.Timedelta(hours=24-i)).strftime("%H:00") for i in range(24)]
        s_vals = np.linspace(1.2, current_slope, 24)
        borehole_strain = np.linspace(0.8, current_slope * 0.72, 24)
        
    # 1. Main Inclinometer Displacement (Alert Orange)
    fig.add_trace(go.Scatter(
        x=x_vals, y=s_vals,
        name="S-07 Inclinometer (mm)",
        line=dict(color="#EA580C" if current_slope <= 15 else "#DC2626", width=3),
        mode="lines+markers"
    ))
    
    # 2. Borehole Strain Trace
    fig.add_trace(go.Scatter(
        x=x_vals, y=borehole_strain,
        name="S-09 Extensometer (mm)",
        line=dict(color="#94A3B8", width=1.5, dash="dot")
    ))
    
    # Critical Geotechnical Slope Failure Threshold Line
    fig.add_hline(
        y=15.0, line_dash="dash", line_color="#DC2626", line_width=1.5,
        annotation_text="Critical Failure Threshold (15mm)", annotation_position="top left",
        annotation_font=dict(size=10, color="#DC2626")
    )
    
    fig.update_layout(
        height=260, margin=dict(l=5, r=5, t=10, b=5),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94A3B8", size=10),
        legend=dict(orientation="h", y=1.2, x=0.0, font=dict(color="#94A3B8")),
        xaxis=dict(showgrid=True, gridcolor="rgba(148, 163, 184, 0.2)"),
        yaxis=dict(title="Displacement (mm)", gridcolor="rgba(148, 163, 184, 0.2)", color="#EA580C"),
        uirevision="varunx_trend_slope"
    )
    return fig


def render_downstream_impact_table(selected_zone: str, risk_level: int, lead_time: float):
    z_info = CATCHMENT_ZONES_DATABASE.get(selected_zone, CATCHMENT_ZONES_DATABASE["Kedarnath Valley (Mandakini Catchment)"])
    wards = z_info.get("vulnerable_wards", [])
    
    pill_classes = ["pill-low", "pill-medium", "pill-high", "pill-critical"]
    pill_labels = ["Low Risk", "Medium Risk", "High Risk", "Critical Risk"]
    current_pill_cls = pill_classes[risk_level]
    current_pill_lbl = pill_labels[risk_level]
    
    rows = []
    for idx, w in enumerate(wards):
        resp_min = max(15, int(lead_time * 60 - w["distance_km"] * 2.5))
        rows.append(
            f"<tr>"
            f"<td style='font-weight:700;color:var(--vx-text);'>{w['name']}</td>"
            f"<td style='color:var(--vx-text-muted);'>{w['lat']:.4f}°N, {w['lon']:.4f}°E</td>"
            f"<td style='color:var(--vx-text-muted);'>{z_info['elevation'] - w['distance_km']*35} m</td>"
            f"<td style='color:var(--vx-text-muted);'>{w['distance_km']} km (Reach {idx+1})</td>"
            f"<td style='font-weight:700;color:var(--vx-accent);'>⏱ {resp_min} min</td>"
            f"<td><span class='risk-pill {current_pill_cls}'>{current_pill_lbl}</span></td>"
            f"<td><span style='background:var(--vx-badge-bg);border:1px solid var(--vx-badge-border);padding:3px 8px;border-radius:6px;font-size:0.75rem;color:var(--vx-badge-color);font-weight:700;'>ACTIVE NODE</span></td>"
            f"</tr>"
        )
    table_rows = "".join(rows)
        
    html = (
        "<div class='varunx-table-wrap'>"
        "<table class='varunx-table'>"
        "<thead>"
        "<tr>"
        "<th>Downstream Ward / Asset</th>"
        "<th>GPS Coordinates</th>"
        "<th>Elevation</th>"
        "<th>Distance</th>"
        "<th>Evacuation Window</th>"
        "<th>Risk Severity</th>"
        "<th>Action</th>"
        "</tr>"
        "</thead>"
        f"<tbody>{table_rows}</tbody>"
        "</table>"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


# Catchment Regional Scenarios Registry (Unique Baselines & Disaster Triggers per Valley)
CATCHMENT_SCENARIOS = {
    "Kedarnath Valley (Mandakini Catchment)": {
        "baseline": {
            "flow_water_level": 3.8, "slope_movement": 2.1, "discharge": 38.0,
            "rainfall_1h": 6.0, "rainfall_24h": 28.0, "air_temp": 8.5, "surface_temp": 7.2
        },
        "disaster_label": "Mandakini Cloudburst",
        "disaster": {
            "flow_water_level": 16.8, "slope_movement": 82.0, "discharge": 380.0,
            "rainfall_1h": 62.0, "rainfall_24h": 210.0, "air_temp": 4.2, "surface_temp": 3.5
        }
    },
    "Teesta River Basin (North Sikkim)": {
        "baseline": {
            "flow_water_level": 4.5, "slope_movement": 3.4, "discharge": 65.0,
            "rainfall_1h": 14.0, "rainfall_24h": 58.0, "air_temp": 14.2, "surface_temp": 13.0
        },
        "disaster_label": "Chungthang Inundation",
        "disaster": {
            "flow_water_level": 18.5, "slope_movement": 55.0, "discharge": 460.0,
            "rainfall_1h": 58.0, "rainfall_24h": 195.0, "air_temp": 11.5, "surface_temp": 10.2
        }
    },
    "Alaknanda Basin (Chamoli & Joshimath Corridor)": {
        "baseline": {
            "flow_water_level": 4.2, "slope_movement": 4.8, "discharge": 52.0,
            "rainfall_1h": 8.0, "rainfall_24h": 36.0, "air_temp": 11.0, "surface_temp": 9.8
        },
        "disaster_label": "Joshimath Subsidence",
        "disaster": {
            "flow_water_level": 13.8, "slope_movement": 110.0, "discharge": 310.0,
            "rainfall_1h": 45.0, "rainfall_24h": 160.0, "air_temp": 6.8, "surface_temp": 5.9
        }
    },
    "Beas River Valley (Kullu-Manali Catchment)": {
        "baseline": {
            "flow_water_level": 3.5, "slope_movement": 1.8, "discharge": 42.0,
            "rainfall_1h": 5.0, "rainfall_24h": 22.0, "air_temp": 16.5, "surface_temp": 15.0
        },
        "disaster_label": "Kullu Monsoon Surge",
        "disaster": {
            "flow_water_level": 15.2, "slope_movement": 38.0, "discharge": 365.0,
            "rainfall_1h": 52.0, "rainfall_24h": 170.0, "air_temp": 12.0, "surface_temp": 11.0
        }
    },
    "Lahaul Valley (Chandra-Bhaga River)": {
        "baseline": {
            "flow_water_level": 2.8, "slope_movement": 1.2, "discharge": 25.0,
            "rainfall_1h": 2.0, "rainfall_24h": 12.0, "air_temp": 5.0, "surface_temp": 3.8
        },
        "disaster_label": "Spiti Mudflow Blockade",
        "disaster": {
            "flow_water_level": 12.2, "slope_movement": 65.0, "discharge": 230.0,
            "rainfall_1h": 35.0, "rainfall_24h": 115.0, "air_temp": 2.5, "surface_temp": 1.8
        }
    }
}


def get_or_create_catchment_state(selected_zone: str) -> dict:
    if "catchments_state" not in st.session_state:
        st.session_state.catchments_state = {}
    if "zone_status" not in st.session_state:
        st.session_state.zone_status = {}

    if selected_zone not in st.session_state.catchments_state:
        scenario = CATCHMENT_SCENARIOS.get(selected_zone, CATCHMENT_SCENARIOS["Kedarnath Valley (Mandakini Catchment)"])
        b_val = scenario["baseline"]
        now = datetime.now()
        warmup = []
        for i in range(8, 0, -1):
            t = (now - pd.Timedelta(seconds=i * 6)).strftime("%H:%M:%S")
            warmup.append({
                "timestamp": t,
                "flow_water_level_m": round(b_val["flow_water_level"] + np.random.uniform(-0.2, 0.2), 2),
                "slope_movement_mm": round(b_val["slope_movement"] + np.random.uniform(-0.2, 0.2), 2),
                "discharge_m3s": round(b_val["discharge"] + np.random.uniform(-2, 2), 1),
                "risk_level": 0
            })
        st.session_state.catchments_state[selected_zone] = {
            **b_val,
            "prev_state_telemetry": None,
            "prev_tick": None,
            "critical_since": None,
            "history": warmup
        }
    return st.session_state.catchments_state[selected_zone]


# Scoped Fragment Container for Telemetry Dashboard (Smooth 10s Auto-Refresh)
@st.fragment(run_every=10)
def render_live_telemetry_fragment(selected_zone: str, auto_stream: bool):
    model, feature_cols = load_or_train_model()
    zstate = get_or_create_catchment_state(selected_zone)
    is_live = st.session_state.get("auto_stream_on", False)

    if is_live:
        sensor_obj, next_st = generate_telemetry_tick(selected_zone, zstate.get("prev_state_telemetry"))
        zstate["prev_state_telemetry"] = next_st
        zstate["flow_water_level"] = sensor_obj.flow_water_level_m
        zstate["slope_movement"] = sensor_obj.slope_movement_mm
        zstate["discharge"] = sensor_obj.discharge_m3s
        zstate["rainfall_1h"] = sensor_obj.rainfall_1h_mm
        zstate["rainfall_24h"] = sensor_obj.rainfall_24h_mm
        zstate["air_temp"] = sensor_obj.air_temp_c
        zstate["surface_temp"] = sensor_obj.surface_temp_c

    flow_water_level = zstate["flow_water_level"]
    slope_movement = zstate["slope_movement"]
    discharge = zstate["discharge"]
    rainfall_1h = zstate["rainfall_1h"]
    rainfall_24h = zstate["rainfall_24h"]
    air_temp = zstate["air_temp"]
    surface_temp = zstate["surface_temp"]

    water_level_rate = 0.08 + (flow_water_level - 5) * 0.03
    tilt_rate = slope_movement * 0.05
    rainfall_3h = rainfall_1h * 2.2

    sensor_input = SensorReading(
        timestamp=datetime.now().strftime("%H:%M:%S"),
        catchment_zone=selected_zone,
        flow_water_level_m=flow_water_level,
        water_level_rate_m_h=water_level_rate,
        slope_movement_mm=slope_movement,
        tilt_rate_mm_h=tilt_rate,
        discharge_m3s=discharge,
        rainfall_1h_mm=rainfall_1h,
        rainfall_3h_mm=rainfall_3h,
        rainfall_24h_mm=rainfall_24h,
        air_temp_c=air_temp,
        surface_temp_c=surface_temp
    )

    X_live = pd.DataFrame([{
        "rainfall_1h_mm": sensor_input.rainfall_1h_mm,
        "rainfall_3h_mm": sensor_input.rainfall_3h_mm,
        "rainfall_24h_mm": sensor_input.rainfall_24h_mm,
        "flow_water_level_m": sensor_input.flow_water_level_m,
        "water_level_rate_m_h": sensor_input.water_level_rate_m_h,
        "slope_movement_mm": sensor_input.slope_movement_mm,
        "tilt_rate_mm_h": sensor_input.tilt_rate_mm_h,
        "discharge_m3s": sensor_input.discharge_m3s,
        "air_temp_c": sensor_input.air_temp_c,
        "surface_temp_c": sensor_input.surface_temp_c
    }])

    risk_level = int(model.predict(X_live)[0])
    risk_proba = model.predict_proba(X_live)[0]
    risk_score = int(risk_proba[risk_level] * 100)
    lead_time = estimate_lead_time(risk_level, water_level_rate, slope_movement)
    risk_label = get_risk_label(risk_level)

    zstate["history"].append({
        "timestamp": sensor_input.timestamp,
        "flow_water_level_m": flow_water_level,
        "slope_movement_mm": slope_movement,
        "discharge_m3s": discharge,
        "risk_level": risk_level
    })
    if len(zstate["history"]) > 35:
        zstate["history"] = zstate["history"][-35:]

    hist_df = pd.DataFrame(zstate["history"])

    if risk_level >= 2 and zstate.get("critical_since") is None:
        zstate["critical_since"] = datetime.now()
    elif risk_level < 2:
        zstate["critical_since"] = None

    if zstate.get("critical_since") is not None:
        elapsed = (datetime.now() - zstate["critical_since"]).total_seconds()
        remaining_seconds = max(0, int(lead_time * 3600 - elapsed))
        countdown_str = format_hms(remaining_seconds)
    else:
        remaining_seconds = None
        countdown_str = None

    st.session_state.zone_status[selected_zone] = risk_label
    current_utc = datetime.utcnow().strftime("%H:%M UTC | %b %d, %Y")
    lead_display = f"⏱ {countdown_str} remaining" if countdown_str else f"{lead_time} Hours"

    # 1. Full-Width Top Alert Banner with Real-Time Client-Side 1s Countdown
    if risk_level == 3:
        b_class = "critical"
        b_title = "🚨 FLASH FLOOD & LANDSLIDE WARNING SYSTEM"
        b_status_html = f'CRITICAL RISK (ALERT LEVEL 3) · Evacuation Window: <span id="varunx-live-timer" class="timer-badge">⏱ {countdown_str} remaining</span>'
    elif risk_level == 2:
        b_class = "high"
        b_title = "⚠️ FLASH FLOOD & LANDSLIDE WARNING SYSTEM"
        b_status_html = f'HIGH RISK (ALERT LEVEL 2) · Evacuation Window: <span id="varunx-live-timer" class="timer-badge">⏱ {countdown_str} remaining</span>'
    elif risk_level == 1:
        b_class = "medium"
        b_title = "🟡 FLASH FLOOD & LANDSLIDE WARNING SYSTEM"
        b_status_html = f"MEDIUM RISK (ALERT LEVEL 1) · Lead Time: {lead_time}h"
    else:
        b_class = "low"
        b_title = "✅ FLASH FLOOD & LANDSLIDE WARNING SYSTEM"
        b_status_html = "LOW RISK (NORMAL BASELINE) · Catchment Stable"

    script_tag = ""
    if remaining_seconds is not None:
        script_tag = f"""
        <script>
        (function(){{
            var serverSec = {remaining_seconds};
            var now = Date.now();
            if (!window.parent._varunxEndTime || Math.abs(window.parent._varunxEndTime - (now + serverSec * 1000)) > 45000) {{
                window.parent._varunxEndTime = now + serverSec * 1000;
            }}
            function tick() {{
                var el = document.getElementById('varunx-live-timer');
                if (!el) return;
                var diff = Math.max(0, Math.floor((window.parent._varunxEndTime - Date.now()) / 1000));
                if (diff <= 0) {{
                    el.innerText = '⏱ 0:00:00 EXPIRED';
                    return;
                }}
                var h = Math.floor(diff / 3600);
                var m = Math.floor((diff % 3600) / 60);
                var s = diff % 60;
                var padM = m < 10 ? '0' + m : m;
                var padS = s < 10 ? '0' + s : s;
                el.innerText = '⏱ ' + h + ':' + padM + ':' + padS + ' remaining';
            }}
            tick();
            if (window.timerInterval) clearInterval(window.timerInterval);
            window.timerInterval = setInterval(tick, 1000);
        }})();
        </script>
        """
    else:
        script_tag = """
        <script>
        window.parent._varunxEndTime = null;
        if (window.timerInterval) clearInterval(window.timerInterval);
        </script>
        """

    banner_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
      body {{ margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: transparent; overflow: hidden; }}
      .alert-banner {{
          padding: 10px 18px;
          border-radius: 8px;
          font-weight: 700;
          font-size: 0.94rem;
          letter-spacing: 0.4px;
          display: flex;
          align-items: center;
          justify-content: space-between;
          box-shadow: 0 4px 14px rgba(234, 88, 12, 0.15);
      }}
      .critical {{ background: linear-gradient(90deg, #DC2626 0%, #EA580C 70%, #C2410C 100%); color: #FFFFFF; border: 1px solid #B91C1C; }}
      .high {{ background: linear-gradient(90deg, #EA580C 0%, #F97316 60%, #FB923C 100%); color: #FFFFFF; border: 1px solid #C2410C; }}
      .medium {{ background: linear-gradient(90deg, #D97706 0%, #F59E0B 100%); color: #FFFFFF; border: 1px solid #B45309; }}
      .low {{ background: linear-gradient(90deg, #059669 0%, #0D9488 100%); color: #FFFFFF; border: 1px solid #047857; }}
      .timer-badge {{
          font-family: monospace;
          font-weight: 800;
          background: rgba(0,0,0,0.3);
          padding: 2px 7px;
          border-radius: 5px;
      }}
    </style>
    </head>
    <body>
      <div class="alert-banner {b_class}">
        <div>
          <span>{b_title}</span> &nbsp;|&nbsp; 
          <span>{b_status_html}</span> &nbsp;|&nbsp; 
          <span style="text-decoration: underline;">{selected_zone.upper()}</span>
        </div>
        <div style="font-size: 0.82rem; opacity: 0.95;">
          {current_utc}
        </div>
      </div>
      {script_tag}
    </body>
    </html>
    """
    st.components.v1.html(banner_html, height=54)

    # 2. Top Row (70% - 30% Width Columns)
    col_3d, col_map = st.columns([7, 3])

    with col_3d:
        st.markdown('<div class="card-header">🏔️ 3D Valley Terrain & Inundation Model</div>', unsafe_allow_html=True)
        render_3d_valley_component(round(flow_water_level, 1), round(slope_movement, 1), selected_zone)
        if slope_movement > 15.0:
            st.error("🚨 Active Geotechnical Slope Failure Scarp Identified on Valley Wall")
        elif flow_water_level > 10.0:
            st.warning("⚠️ Severe River Channel Inundation Threat Detected")
        else:
            st.success("✅ Canyon river flow & valley slope displacement within stable baseline")

    with col_map:
        st.markdown('<div class="card-header">🗺️ Vulnerable Zones & Asset Map (Carto Dark)</div>', unsafe_allow_html=True)
        geomap_fig = create_catchment_geomap(selected_zone, risk_level, flow_water_level, slope_movement)
        st.plotly_chart(geomap_fig, use_container_width=True, key="catchment_geomap_chart", config={"displayModeBar": False})

    # 3. Middle Row: 4 Metric Cards + Risk Probability Progress
    prev = zstate.get("prev_tick")
    def _delta(curr, key, suffix=""):
        if prev is None:
            return None
        d = curr - prev.get(key, curr)
        if abs(d) < 1e-6:
            return None
        arrow = "▲" if d > 0 else "▼"
        return f"{arrow} {abs(d):.1f}{suffix}"

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🌧️ Rainfall (mm/h)", f"{rainfall_1h:.0f} mm/h", delta=f"24h: {rainfall_24h:.0f}mm")
    m2.metric("🌊 River Stage (m)", f"{flow_water_level:.1f} m", delta=_delta(flow_water_level, "flow_water_level", " m"))
    m3.metric("⛰️ Slope Movement (mm)", f"{slope_movement:.1f} mm", delta=_delta(slope_movement, "slope_movement", " mm"), delta_color="inverse")
    m4.metric("⚡ Discharge (m³/s)", f"{discharge:.0f} m³/s", delta=_delta(discharge, "discharge", " m³/s"))

    zstate["prev_tick"] = {
        "flow_water_level": flow_water_level,
        "slope_movement": slope_movement,
        "discharge": discharge
    }

    # Horizontal 4-Tier Risk Probability Bar Breakdown
    st.markdown('<div class="card-header" style="margin-top:0.8rem;">🎯 ML Classifier Risk Probability Breakdown</div>', unsafe_allow_html=True)
    p_cols = st.columns(4)
    tiers = [
        ("Low Risk", risk_proba[0], "#16A34A"),
        ("Medium Risk", risk_proba[1], "#D97706"),
        ("High Risk", risk_proba[2], "#EA580C"),
        ("Critical Risk", risk_proba[3], "#DC2626")
    ]
    for c, (t_lbl, t_prob, t_col) in zip(p_cols, tiers):
        c.markdown(f"""
        <div style="background:var(--vx-card-bg);border:1px solid var(--vx-border);border-radius:10px;padding:0.75rem;text-align:center;box-shadow:0 2px 8px var(--vx-shadow);">
            <div style="font-size:0.78rem;font-weight:700;color:var(--vx-text-muted);text-transform:uppercase;">{t_lbl}</div>
            <div style="font-size:1.4rem;font-weight:800;color:{t_col};margin:0.2rem 0;">{t_prob*100:.1f}%</div>
            <div style="background:var(--vx-border-subtle);height:6px;border-radius:999px;overflow:hidden;border:1px solid var(--vx-border);">
                <div style="background:{t_col};width:{t_prob*100:.1f}%;height:100%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 4. Bottom Row: Time Series Trends (50% - 50% Columns)
    col_r1, col_r2 = st.columns(2)

    with col_r1:
        st.markdown('<div class="card-header">📈 Rainfall & River Stage (24HR Trend)</div>', unsafe_allow_html=True)
        rain_fig = create_rainfall_river_stage_trend(hist_df, risk_level, flow_water_level)
        st.plotly_chart(rain_fig, use_container_width=True, key="rain_river_stage_chart", config={"displayModeBar": False})

    with col_r2:
        st.markdown('<div class="card-header">📉 Slope Displacement & Ground Movement (24HR Trend)</div>', unsafe_allow_html=True)
        slope_fig = create_slope_displacement_trend(hist_df, slope_movement)
        st.plotly_chart(slope_fig, use_container_width=True, key="slope_displacement_chart", config={"displayModeBar": False})

    # 5. Downstream Impact & Vulnerable Ward Inventory (Image 1 Style)
    st.markdown('<div class="card-header" style="margin-top:1.2rem;">🏘️ Downstream Vulnerable Ward Impact & Response Inventory</div>', unsafe_allow_html=True)
    render_downstream_impact_table(selected_zone, risk_level, lead_time)

    # 6. Emergency Alert Gateway Action (NDRF / CAP XML)
    if risk_level >= 2:
        st.markdown("---")
        st.error(f"🚨 **NDRF Emergency Protocol Triggered** | Evacuation Window: **{lead_display}**")
        if st.button("📢 Broadcast VarunX Emergency Alert (NDRF/SDMA/CAP XML)", use_container_width=True, key="broadcast_btn"):
            trigger_reason = f"24h Rainfall {rainfall_24h:.0f}mm, Slope Displacement {slope_movement:.1f}mm, Flow {flow_water_level:.1f}m"
            alert = dispatch_emergency_alert(
                catchment_zone=selected_zone,
                risk_level=risk_level,
                risk_label=risk_label,
                lead_time_hours=lead_time,
                trigger_reason=trigger_reason
            )
            st.success(f"✅ Emergency Alert Broadcast Dispatched! ID: {alert.alert_id}")

            prev_col, xml_col = st.columns([1, 1.3])
            with prev_col:
                st.markdown("**📱 Public SMS Preview**")
                render_sms_preview(selected_zone, risk_label, lead_time)
            with xml_col:
                st.markdown("**📄 ITU CAP v1.2 XML Payload (NDMA Sachet-Compatible)**")
                st.code(build_cap_xml_preview(selected_zone, risk_label, lead_time, trigger_reason, alert.alert_id), language="xml")


# TAB 1: Live Monitoring Page View
def page_live_monitoring():
    st.sidebar.markdown("### 🎛️ Catchment Controls")
    catchment_options = list(CATCHMENT_ZONES_DATABASE.keys())
    
    if "live_catchment_zone" not in st.session_state:
        st.session_state.live_catchment_zone = catchment_options[0]

    selected_zone = st.sidebar.selectbox(
        "Target Hilly Catchment / Ward", catchment_options, key="live_catchment_zone"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📡 Telemetry Configuration")
    auto_stream = st.sidebar.toggle("⚡ Auto-Stream Live Telemetry", value=False, key="auto_stream_on")
    
    zstate = get_or_create_catchment_state(selected_zone)
    scenario_info = CATCHMENT_SCENARIOS.get(selected_zone, CATCHMENT_SCENARIOS["Kedarnath Valley (Mandakini Catchment)"])

    # Ensure widget session state exists for the selected zone
    if f"s_rf1_{selected_zone}" not in st.session_state:
        st.session_state[f"s_rf1_{selected_zone}"] = float(zstate["rainfall_1h"])
        st.session_state[f"s_rf24_{selected_zone}"] = float(zstate["rainfall_24h"])
        st.session_state[f"s_atemp_{selected_zone}"] = float(zstate["air_temp"])
        st.session_state[f"s_stemp_{selected_zone}"] = float(zstate["surface_temp"])
        st.session_state[f"s_flow_{selected_zone}"] = float(zstate["flow_water_level"])
        st.session_state[f"s_slope_{selected_zone}"] = float(zstate["slope_movement"])
        st.session_state[f"s_disch_{selected_zone}"] = float(zstate["discharge"])

    with st.sidebar.expander("🌧️ Rainfall & Weather Sliders", expanded=True):
        rainfall_1h = st.slider("1-Hour Rainfall (mm/h)", 0.0, 80.0, step=1.0, key=f"s_rf1_{selected_zone}")
        rainfall_24h = st.slider("24-Hour Rainfall (mm)", 0.0, 250.0, step=2.0, key=f"s_rf24_{selected_zone}")
        air_temp = st.slider("Air Temp (°C)", -10.0, 30.0, step=0.5, key=f"s_atemp_{selected_zone}")
        surface_temp = st.slider("Surface Temp (°C)", -10.0, 30.0, step=0.5, key=f"s_stemp_{selected_zone}")

    with st.sidebar.expander("🌊 Hydrological & Geotechnical Sliders", expanded=True):
        flow_water_level = st.slider("River Flow Level (m)", 0.5, 20.0, step=0.2, key=f"s_flow_{selected_zone}")
        slope_movement = st.slider("Slope Displacement (mm)", 0.0, 150.0, step=0.5, key=f"s_slope_{selected_zone}")
        discharge = st.slider("Discharge Rate (m³/s)", 5.0, 500.0, step=5.0, key=f"s_disch_{selected_zone}")

    if not auto_stream:
        zstate["flow_water_level"] = flow_water_level
        zstate["slope_movement"] = slope_movement
        zstate["discharge"] = discharge
        zstate["rainfall_1h"] = rainfall_1h
        zstate["rainfall_24h"] = rainfall_24h
        zstate["air_temp"] = air_temp
        zstate["surface_temp"] = surface_temp

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚡ Quick Presets")
    c_p1, c_p2 = st.sidebar.columns(2)
    
    def _apply_preset(zone: str, is_disaster: bool):
        sc = CATCHMENT_SCENARIOS.get(zone, CATCHMENT_SCENARIOS["Kedarnath Valley (Mandakini Catchment)"])
        vals = sc["disaster"] if is_disaster else sc["baseline"]
        zs = get_or_create_catchment_state(zone)
        for k, v in vals.items():
            zs[k] = v
        st.session_state[f"s_flow_{zone}"] = float(vals["flow_water_level"])
        st.session_state[f"s_slope_{zone}"] = float(vals["slope_movement"])
        st.session_state[f"s_disch_{zone}"] = float(vals["discharge"])
        st.session_state[f"s_rf1_{zone}"] = float(vals["rainfall_1h"])
        st.session_state[f"s_rf24_{zone}"] = float(vals["rainfall_24h"])
        st.session_state[f"s_atemp_{zone}"] = float(vals["air_temp"])
        st.session_state[f"s_stemp_{zone}"] = float(vals["surface_temp"])

    c_p1.button("✅ Baseline", use_container_width=True, on_click=_apply_preset, args=(selected_zone, False))
    c_p2.button(f"🚨 {scenario_info['disaster_label']}", use_container_width=True, on_click=_apply_preset, args=(selected_zone, True))

    # Render scoped fragment container for flicker-free plot updates
    render_live_telemetry_fragment(selected_zone, auto_stream)


# TAB 2: Interactive GIS Map
def page_gis_map():
    st.markdown("## 🗺️ VarunX Hilly Region Spatial Catchment Map")
    st.info("Maps vulnerable hilly catchment zones, real-time risk markers, and downstream village ward threat vectors.")
    
    col_l, col_r = st.columns([3, 1])
    with col_r:
        selected_zone_gis = st.selectbox("Select Catchment Area", list(CATCHMENT_ZONES_DATABASE.keys()))
        sim_risk = st.select_slider("Simulated Threat Level", options=[0, 1, 2, 3], format_func=lambda x: ["LOW", "MEDIUM", "HIGH", "CRITICAL"][x])
        
        zone_data = CATCHMENT_ZONES_DATABASE[selected_zone_gis]
        st.markdown("### Catchment Metadata")
        st.write(f"**Elevation:** {zone_data['elevation']} m")
        st.write(f"**Location:** {zone_data['district']}, {zone_data['state']}")
        st.write(f"**Coordinates:** {zone_data['lat']}° N, {zone_data['lon']}° E")
        
        st.markdown("### Downstream Wards")
        total_exposed = sum(w["population"] for w in zone_data["vulnerable_wards"])
        for ward in zone_data["vulnerable_wards"]:
            st.write(f"• **{ward['name']}** ({ward['distance_km']} km) — *Pop: {ward['population']:,}*")

    with col_l:
        exposure_color = "#DC2626" if sim_risk >= 2 else ("#D97706" if sim_risk == 1 else "#16A34A")
        st.markdown(
            f"""<div style="background:var(--vx-card-bg);border:1px solid var(--vx-border);border-left:5px solid {exposure_color};
            border-radius:10px;padding:0.9rem 1.2rem;margin-bottom:0.8rem;box-shadow:0 2px 8px var(--vx-shadow);">
            <span style="color:var(--vx-text-muted);font-size:0.85rem;font-weight:700;letter-spacing:0.4px;">DOWNSTREAM POPULATION EXPOSED</span><br>
            <span style="color:var(--vx-accent);font-size:1.9rem;font-weight:800;">{total_exposed:,} residents</span>
            <span style="color:var(--vx-text-muted);font-size:0.85rem;"> across {len(zone_data['vulnerable_wards'])} vulnerable wards</span>
            </div>""",
            unsafe_allow_html=True
        )
        gis_fig = create_plotly_gis_map(selected_zone_gis, sim_risk, 12.0, 120.0)
        st.plotly_chart(gis_fig, use_container_width=True, key="gis_map_chart")


# TAB 3: Emergency Alert Gateway
def page_alert_gateway():
    st.markdown("## 🚨 VarunX Emergency Alert Gateway (NDRF & DM Division)")
    st.caption("Generates ITU CAP v1.2-compliant XML payloads for plug-and-play integration with the NDMA Sachet portal, alongside multi-channel SMS/Email/Webhook dispatch.")
    
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.subheader("📢 Broadcast Manual Test Warning")
        zone_name = st.selectbox("Target Hilly Catchment", list(CATCHMENT_ZONES_DATABASE.keys()))
        risk_lvl = st.selectbox("Severity Level", [3, 2, 1], format_func=lambda x: f"Level {x}: {['LOW','MEDIUM','HIGH','CRITICAL'][x]}")
        lead_t = st.number_input("Lead Time Window (Hours)", min_value=0.5, max_value=48.0, value=3.0, step=0.5)
        channels = st.multiselect("Active Notification Channels", ["SMS", "Email", "Webhook", "NDMA_CAP_XML"], default=["SMS", "Email", "Webhook", "NDMA_CAP_XML"])
        webhook_url = st.text_input("Custom Webhook / Telegram Bot Endpoint (Optional)", value="")
        
        if st.button("🚀 Dispatch Emergency Warning Broadcast Now", use_container_width=True):
            alert = dispatch_emergency_alert(
                catchment_zone=zone_name,
                risk_level=risk_lvl,
                risk_label=["LOW", "MEDIUM", "HIGH", "CRITICAL"][risk_lvl],
                lead_time_hours=lead_t,
                trigger_reason="NDRF Control Room Manual Emergency Simulation Test",
                channels=channels,
                webhook_url=webhook_url if webhook_url.strip() else None
            )
            st.success(f"Emergency Warning Dispatched! Reference ID: `{alert.alert_id}`")
            st.code(alert.message_body, language="markdown")

            prev_col, xml_col = st.columns([1, 1.3])
            with prev_col:
                st.markdown("**📱 Public SMS Preview**")
                render_sms_preview(zone_name, ["LOW", "MEDIUM", "HIGH", "CRITICAL"][risk_lvl], lead_t)
            with xml_col:
                st.markdown("**📄 ITU CAP v1.2 XML Payload (NDMA Sachet-Compatible)**")
                st.code(
                    build_cap_xml_preview(
                        zone_name, ["LOW", "MEDIUM", "HIGH", "CRITICAL"][risk_lvl],
                        lead_t, "NDRF Control Room Manual Emergency Simulation Test", alert.alert_id
                    ),
                    language="xml"
                )
            
    with col2:
        st.subheader("📋 Dispatched Warning Audit Log")
        history = load_alert_history()
        if history:
            df_hist = pd.DataFrame(history)
            st.dataframe(df_hist[["timestamp", "alert_id", "catchment_zone", "risk_label", "lead_time_hours", "status"]], use_container_width=True, height=350)
        else:
            st.info("No dispatched warnings recorded yet.")


# TAB 4: ML Risk Engine Studio
def page_ml_studio():
    st.markdown("## 📊 VarunX Machine Learning Risk Engine Diagnostics")
    st.caption("Inspect classifier metrics, feature importance rankings, confusion matrix, and retrain options.")

    st.markdown(
        f"""<div style="background:var(--vx-card-bg);border:1px solid var(--vx-border);border-radius:10px;
        padding:0.9rem 1.2rem;margin-bottom:1rem;box-shadow:0 2px 8px var(--vx-shadow);">
        <span style="color:var(--vx-text-muted);font-size:0.85rem;font-weight:700;">10 LIVE MULTI-SOURCE PARAMETERS EVALUATED PER TICK</span><br>
        <span style="color:var(--vx-text);font-size:0.88rem;font-weight:600;">{" · ".join(VARUNX_FEATURE_COLS)}</span>
        </div>""",
        unsafe_allow_html=True
    )
    
    metrics_path = "models/metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
            
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Overall Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
        c2.metric("ROC-AUC Score", f"{metrics['roc_auc']:.4f}")
        c3.metric("Training Samples", f"{metrics.get('train_samples', 2000):,}")
        c4.metric("Last Retrained", metrics.get("trained_at", "Recent"))
        
        st.markdown("---")
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.subheader("🏆 Feature Importance Ranking")
            fi_df = pd.DataFrame(metrics["feature_importances"])
            fig_fi = px.bar(fi_df, x="importance", y="feature", orientation="h", color="importance", color_continuous_scale="Viridis")
            fig_fi.update_layout(
                height=340, yaxis=dict(autorange="reversed"), showlegend=False,
                paper_bgcolor="#0f172a", plot_bgcolor="#0f172a", font=dict(color="#f8fafc")
            )
            st.plotly_chart(fig_fi, use_container_width=True, key="feature_importance_chart")
            
        with col_m2:
            st.subheader("🧩 Confusion Matrix")
            cm = np.array(metrics["confusion_matrix"])
            labels = ["Low", "Medium", "High", "Critical"]
            fig_cm = px.imshow(cm, x=labels, y=labels, text_auto=True, color_continuous_scale="Reds")
            fig_cm.update_layout(
                height=340, xaxis_title="Predicted Class", yaxis_title="Actual Class",
                paper_bgcolor="#0f172a", plot_bgcolor="#0f172a", font=dict(color="#f8fafc")
            )
            st.plotly_chart(fig_cm, use_container_width=True, key="confusion_matrix_chart")

    else:
        st.info("Model metrics document `models/metrics.json` available.")
        
    st.markdown("---")
    st.markdown("### 🛰️ Operational Retraining Roadmap")
    st.info(
        "For field deployment, VarunX retrains on the **ISRO NRSC Landslide Atlas** "
        "(~80,000 catalogued historical Indian landslides) combined with **IMD gridded rainfall data**, "
        "extending beyond this synthetic training set. See `mlTraining.txt` for the external GPU training reference."
    )


# TAB 5: Historical Replay & Vulnerable Zone Inventory
def page_historical_and_inventory():
    st.markdown("## 📚 Historical Events & Vulnerable Zone Inventory")
    tab1, tab2 = st.tabs(["🔄 Historical Event Replay", "🗺️ Vulnerable Zone Inventory"])
    
    with tab1:
        st.subheader("Past Flash Flood & Landslide Events Database")
        st.info("Click 'Replay Event' to instantly load past disaster conditions and navigate into the Live Monitoring display.")
        
        df = load_historical_events()
        if df is not None:
            st.dataframe(df.head(20), use_container_width=True, height=280)
            
            c_r1, c_r2 = st.columns([2, 1])
            with c_r1:
                selected_event = st.selectbox("Select Historical Event to Replay", [
                    "2023 Sikkim Teesta Flash Flood (Extreme Cloudburst)",
                    "2013 Kedarnath Mandakini Flash Flood (Multi-day Torrential Rain)",
                    "2021 Chamoli Flash Flood & Debris Surge",
                    "2020 Lahaul Spiti Landslide & River Blockade"
                ])
            def _replay_event(event_name: str):
                if "Kedarnath" in event_name:
                    target_zone = "Kedarnath Valley (Mandakini Catchment)"
                    r_vals = {"flow_water_level": 16.5, "slope_movement": 85.0, "discharge": 340.0, "rainfall_1h": 55.0, "rainfall_24h": 190.0, "air_temp": 4.0, "surface_temp": 3.2}
                elif "Sikkim" in event_name:
                    target_zone = "Teesta River Basin (North Sikkim)"
                    r_vals = {"flow_water_level": 18.2, "slope_movement": 52.0, "discharge": 450.0, "rainfall_1h": 58.0, "rainfall_24h": 190.0, "air_temp": 11.0, "surface_temp": 9.8}
                elif "Chamoli" in event_name:
                    target_zone = "Alaknanda Basin (Chamoli & Joshimath Corridor)"
                    r_vals = {"flow_water_level": 14.5, "slope_movement": 98.0, "discharge": 320.0, "rainfall_1h": 46.0, "rainfall_24h": 165.0, "air_temp": 6.5, "surface_temp": 5.5}
                else:
                    target_zone = "Lahaul Valley (Chandra-Bhaga River)"
                    r_vals = {"flow_water_level": 12.0, "slope_movement": 55.0, "discharge": 220.0, "rainfall_1h": 34.0, "rainfall_24h": 110.0, "air_temp": 2.2, "surface_temp": 1.5}
                
                zs = get_or_create_catchment_state(target_zone)
                for k, v in r_vals.items():
                    zs[k] = v
                st.session_state["live_catchment_zone"] = target_zone
                st.session_state[f"s_flow_{target_zone}"] = float(r_vals["flow_water_level"])
                st.session_state[f"s_slope_{target_zone}"] = float(r_vals["slope_movement"])
                st.session_state[f"s_disch_{target_zone}"] = float(r_vals["discharge"])
                st.session_state[f"s_rf1_{target_zone}"] = float(r_vals["rainfall_1h"])
                st.session_state[f"s_rf24_{target_zone}"] = float(r_vals["rainfall_24h"])
                st.session_state[f"s_atemp_{target_zone}"] = float(r_vals["air_temp"])
                st.session_state[f"s_stemp_{target_zone}"] = float(r_vals["surface_temp"])
                # Automatically navigate to Live Monitoring view
                st.session_state["nav_page"] = "🔴 Live Monitoring & 3D Valley"

            with c_r2:
                st.markdown("#### ")
                st.button("▶️ Replay Event Conditions into Live Dashboard", use_container_width=True, on_click=_replay_event, args=(selected_event,))
        else:
            st.error("Historical event data file not found.")

    with tab2:
        st.subheader("Vulnerable Catchment Zone Inventory")
        st.caption("Search known vulnerable hilly zones and hand-off directly into Live Monitoring.")
        
        inv_data = []
        for name, info in CATCHMENT_ZONES_DATABASE.items():
            inv_data.append({
                "Zone / Catchment": name,
                "State": info["state"],
                "District": info["district"],
                "Elevation (m)": info["elevation"],
                "Latitude": info["lat"],
                "Longitude": info["lon"],
                "Downstream Wards Count": len(info["vulnerable_wards"])
            })
        inv_df = pd.DataFrame(inv_data)
        st.dataframe(inv_df, use_container_width=True)
        
        c_i1, c_i2 = st.columns([2, 1])
        with c_i1:
            target_inv_zone = st.selectbox("Select Zone for Direct Live Monitoring Hand-off", list(CATCHMENT_ZONES_DATABASE.keys()))
        
        def _handoff_zone(zone_name: str):
            st.session_state["live_catchment_zone"] = zone_name
            st.session_state["nav_page"] = "🔴 Live Monitoring & 3D Valley"
            
        with c_i2:
            st.markdown("#### ")
            st.button("🎯 Hand-off Zone to Live Monitoring", use_container_width=True, on_click=_handoff_zone, args=(target_inv_zone,))


def main():
    inject_custom_css()

    is_live = st.session_state.get("auto_stream_on", False)
    dot_class = "live-pulse-dot" if is_live else "live-pulse-dot offline"
    live_label = "LIVE TELEMETRY STREAMING (10s Pulse Engine)" if is_live else "TELEMETRY IDLE (Manual Control Mode)"

    # Header Card
    st.markdown(f"""
    <div style="background:var(--vx-card-bg);border:1px solid var(--vx-border);border-radius:12px;padding:1.1rem 1.4rem;margin-bottom:1rem;display:flex;align-items:center;justify-content:space-between;box-shadow:0 4px 14px var(--vx-shadow);">
        <div>
            <div style="font-size:1.65rem;font-weight:800;letter-spacing:-0.4px;color:var(--vx-accent);display:flex;align-items:center;gap:0.6rem;">
                🏔️ VarunX Early Warning System
            </div>
            <div style="font-size:0.85rem;color:var(--vx-text-muted);margin-top:0.2rem;font-weight:600;">
                AI & IoT-Powered Flash Flood & Landslide Risk Prediction for Hilly Regions | SIH26192 • Ministry of Home Affairs (NDRF, DM Division)
            </div>
        </div>
        <div style="background:var(--vx-badge-bg);border:1px solid var(--vx-badge-border);border-radius:8px;padding:0.45rem 0.9rem;display:flex;align-items:center;gap:0.55rem;">
            <span class="{dot_class}"></span>
            <span style="font-size:0.82rem;font-weight:700;letter-spacing:0.4px;color:var(--vx-accent);">{live_label}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    nav_options = [
        "🔴 Live Monitoring & 3D Valley",
        "🗺️ Interactive GIS Risk Map",
        "🚨 Emergency Alert Gateway",
        "📊 ML Risk Engine Studio",
        "📚 Historical Replay & Inventory"
    ]
    if "nav_page" not in st.session_state:
        st.session_state.nav_page = nav_options[0]

    page = st.sidebar.radio(
        "VarunX Navigation",
        nav_options,
        key="nav_page"
    )
    st.sidebar.markdown("---")
    
    if page == "🔴 Live Monitoring & 3D Valley":
        page_live_monitoring()
    elif page == "🗺️ Interactive GIS Risk Map":
        page_gis_map()
    elif page == "🚨 Emergency Alert Gateway":
        page_alert_gateway()
    elif page == "📊 ML Risk Engine Studio":
        page_ml_studio()
    else:
        page_historical_and_inventory()


if __name__ == "__main__":
    main()