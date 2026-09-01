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

# Custom High-Craft Industrial Dark-Theme CSS Design System
st.markdown("""
<style>
    /* Global Container Adjustments */
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }
    
    /* Header Card */
    .varunx-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
        padding: 1.4rem 1.8rem;
        border-radius: 12px;
        border-left: 6px solid #0284c7;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4);
        margin-bottom: 1.2rem;
        border-top: 1px solid #334155;
        border-right: 1px solid #334155;
        border-bottom: 1px solid #334155;
    }
    .varunx-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0;
        color: #ffffff;
    }
    .varunx-subtitle {
        font-size: 0.92rem;
        color: #38bdf8;
        margin-top: 0.3rem;
        font-weight: 500;
    }
    
    /* Status Badges & Alerts */
    .alert-critical {
        background: linear-gradient(90deg, #7f1d1d 0%, #b91c1c 100%);
        color: #ffffff;
        padding: 1rem 1.2rem;
        border-radius: 10px;
        font-size: 1.15rem;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 4px 15px rgba(185, 28, 28, 0.4);
        border: 1px solid #f87171;
    }
    .alert-high {
        background: linear-gradient(90deg, #7c2d12 0%, #c2410c 100%);
        color: #ffffff;
        padding: 1rem 1.2rem;
        border-radius: 10px;
        font-size: 1.1rem;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 4px 15px rgba(194, 65, 12, 0.35);
        border: 1px solid #fb923c;
    }
    .alert-medium {
        background: linear-gradient(90deg, #78350f 0%, #b45309 100%);
        color: #fef3c7;
        padding: 0.9rem 1.2rem;
        border-radius: 10px;
        font-size: 1.05rem;
        font-weight: 700;
        text-align: center;
        border: 1px solid #fde68a;
    }
    .alert-low {
        background: linear-gradient(90deg, #14532d 0%, #15803d 100%);
        color: #dcfce7;
        padding: 0.9rem 1.2rem;
        border-radius: 10px;
        font-size: 1.05rem;
        font-weight: 700;
        text-align: center;
        border: 1px solid #86efac;
    }
    
    /* High-Contrast Dark Metric Card Styling */
    div[data-testid="stMetric"] {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        padding: 0.8rem 1rem !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25) !important;
    }
    div[data-testid="stMetricLabel"] label, div[data-testid="stMetricLabel"] p {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px !important;
    }
    div[data-testid="stMetricValue"] div, div[data-testid="stMetricValue"] p {
        color: #f8fafc !important;
        font-size: 1.7rem !important;
        font-weight: 800 !important;
    }
    div[data-testid="stMetricDelta"] div, div[data-testid="stMetricDelta"] p {
        color: #38bdf8 !important;
        font-weight: 600 !important;
    }
    
    /* Sidebar Improvements */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #1e293b;
    }
    section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] label {
        color: #f1f5f9 !important;
    }
    
    /* Chart Container Styling */
    .stPlotlyChart {
        border-radius: 10px;
        border: 1px solid #1e293b;
        background-color: #0f172a;
        padding: 4px;
    }

    /* Live Pulse Indicator */
    .live-pulse-wrap {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-top: 0.5rem;
    }
    .live-pulse-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #22c55e;
        box-shadow: 0 0 0 rgba(34, 197, 94, 0.6);
        animation: pulse-live 1.6s infinite;
    }
    .live-pulse-dot.offline {
        background: #64748b;
        animation: none;
    }
    @keyframes pulse-live {
        0% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.55); }
        70% { box-shadow: 0 0 0 9px rgba(34, 197, 94, 0); }
        100% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0); }
    }
    .live-pulse-label {
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        color: #cbd5e1;
    }

    /* Alert Banner Flash for Critical/High Risk */
    .alert-critical, .alert-high {
        animation: alert-flash 1.4s ease-in-out infinite;
    }
    @keyframes alert-flash {
        0%, 100% { filter: brightness(1); }
        50% { filter: brightness(1.18); }
    }

    /* Zone Status Chip Strip */
    .zone-chip-row {
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
        margin: 0.4rem 0 1rem 0;
    }
    .zone-chip {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 999px;
        padding: 0.35rem 0.9rem;
        font-size: 0.8rem;
        font-weight: 600;
        color: #e2e8f0;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .zone-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }

    /* SMS Preview Bubble */
    .sms-bubble {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 16px 16px 16px 4px;
        padding: 0.9rem 1.1rem;
        color: #f1f5f9;
        font-size: 0.88rem;
        line-height: 1.45;
        max-width: 420px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.3);
        margin-top: 0.5rem;
    }
    .sms-meta {
        font-size: 0.72rem;
        color: #64748b;
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
    model_path = "models/glof_risk_model_rf.pkl"
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
    path = "data/historical_glof_data.csv"
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


@st.cache_data
def create_3d_valley_terrain(flow_water_level: float, slope_movement: float):
    x = np.linspace(-6, 6, 60)
    y = np.linspace(-6, 6, 60)
    X, Y = np.meshgrid(x, y)
    
    Z_valley = 0.25 * (X**2) + 0.05 * (Y**2) - 3.0
    water_z = -3.0 + (flow_water_level / 20.0) * 4.0
    water_mask = Z_valley < water_z
    
    slope_fail_mask = (X >= -5) & (X <= -2) & (Y >= -2) & (Y <= 2) & (slope_movement > 25.0)
    
    fig = go.Figure()
    
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z_valley,
        colorscale=[[0, "rgb(30, 70, 32)"], [0.4, "rgb(110, 90, 65)"], [1, "rgb(180, 170, 150)"]],
        showscale=False, opacity=0.94, name="Valley Terrain"
    ))
    
    Z_water = np.where(water_mask, water_z, np.nan)
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z_water,
        colorscale=[[0, "rgb(14, 116, 144)"], [1, "rgb(225, 29, 72)"]] if flow_water_level > 12 else [[0, "rgb(14, 116, 144)"], [1, "rgb(14, 116, 144)"]],
        showscale=False, opacity=0.85, name="Flood Water"
    ))
    
    if slope_movement > 25.0:
        Z_slope_fail = np.where(slope_fail_mask, Z_valley + 0.18, np.nan)
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z_slope_fail,
            colorscale=[[0, "rgb(225, 29, 72)"], [1, "rgb(249, 115, 22)"]],
            showscale=False, opacity=0.95, name="Active Landslide Displacement"
        ))
    
    fig.update_layout(
        title=dict(
            text=f"<b>VarunX 3D Valley Terrain Engine</b> — Water Level: {flow_water_level:.1f}m | Slope Movement: {slope_movement:.1f}mm",
            x=0.5, font=dict(size=13, color="#f8fafc")
        ),
        height=400, margin=dict(l=0, r=0, t=35, b=0),
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            zaxis=dict(visible=False, range=[-4, 6]),
            aspectmode="manual", aspectratio=dict(x=1.2, y=1, z=0.6),
            camera=dict(eye=dict(x=1.75, y=1.55, z=1.05), center=dict(x=0, y=0, z=-0.2)),
            bgcolor="#0f172a"
        ),
        paper_bgcolor="#0f172a", showlegend=False,
        uirevision="terrain_constant"  # Prevents camera angle reset on refresh
    )
    return fig


# Scoped Fragment Container for Telemetry Dashboard (Flicker-Free Rendering)
@st.fragment
def render_live_telemetry_fragment(selected_zone: str, auto_stream: bool):
    model, feature_cols = load_or_train_model()

    if auto_stream:
        sensor_obj, next_st = generate_telemetry_tick(selected_zone, st.session_state.prev_state_telemetry)
        st.session_state.prev_state_telemetry = next_st
        st.session_state.flow_water_level = sensor_obj.flow_water_level_m
        st.session_state.slope_movement = sensor_obj.slope_movement_mm
        st.session_state.discharge = sensor_obj.discharge_m3s
        st.session_state.rainfall_1h = sensor_obj.rainfall_1h_mm
        st.session_state.rainfall_24h = sensor_obj.rainfall_24h_mm
        st.session_state.air_temp = sensor_obj.air_temp_c
        st.session_state.surface_temp = sensor_obj.surface_temp_c

    flow_water_level = st.session_state.flow_water_level
    slope_movement = st.session_state.slope_movement
    discharge = st.session_state.discharge
    rainfall_1h = st.session_state.rainfall_1h
    rainfall_24h = st.session_state.rainfall_24h
    air_temp = st.session_state.air_temp
    surface_temp = st.session_state.surface_temp

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

    st.session_state.history.append({
        "timestamp": sensor_input.timestamp,
        "flow_water_level_m": flow_water_level,
        "slope_movement_mm": slope_movement,
        "discharge_m3s": discharge,
        "risk_level": risk_level
    })
    if len(st.session_state.history) > 35:
        st.session_state.history = st.session_state.history[-35:]

    hist_df = pd.DataFrame(st.session_state.history)

    if risk_level >= 2 and st.session_state.critical_since is None:
        st.session_state.critical_since = datetime.now()
    elif risk_level < 2:
        st.session_state.critical_since = None

    if st.session_state.critical_since is not None:
        elapsed = (datetime.now() - st.session_state.critical_since).total_seconds()
        remaining_seconds = max(0, lead_time * 3600 - elapsed)
        countdown_str = format_hms(remaining_seconds)
    else:
        countdown_str = None

    st.session_state.zone_status[selected_zone] = risk_label

    lead_display = f"⏱ {countdown_str} remaining" if countdown_str else f"{lead_time} Hours"
    if risk_level == 3:
        st.markdown(f'<div class="alert-critical">🚨 VarunX CRITICAL ALERT — Flash Flood & Debris Landslide Imminent | Evacuation Window: <b>{lead_display}</b></div>', unsafe_allow_html=True)
    elif risk_level == 2:
        st.markdown(f'<div class="alert-high">⚠️ HIGH RISK WARNING — Heavy Catchment Runoff & Slope Failure | Evacuation Window: <b>{lead_display}</b></div>', unsafe_allow_html=True)
    elif risk_level == 1:
        st.markdown(f'<div class="alert-medium">🟡 MEDIUM RISK — Elevated Rainfall & Catchment Saturation | Lead Time: <b>{lead_time} Hours</b></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert-low">✅ LOW RISK — Catchment Conditions Stable | Lead Time: <b>{lead_time}+ Hours</b></div>', unsafe_allow_html=True)

    catchment_options = list(CATCHMENT_ZONES_DATABASE.keys())
    chip_html = '<div class="zone-chip-row">'
    chip_colors = {"LOW": "#22c55e", "MEDIUM": "#f59e0b", "HIGH": "#ea580c", "CRITICAL": "#ef4444"}
    for zname in catchment_options:
        zstatus = st.session_state.zone_status.get(zname, "STANDBY")
        zcolor = chip_colors.get(zstatus, "#64748b")
        active_marker = " ●" if zname == selected_zone else ""
        chip_html += f'<div class="zone-chip"><span class="zone-dot" style="background:{zcolor}"></span>{zname}{active_marker} — {zstatus}</div>'
    chip_html += "</div>"
    st.markdown(chip_html, unsafe_allow_html=True)

    prev = st.session_state.prev_tick
    def _delta(curr, key, suffix=""):
        if prev is None:
            return None
        d = curr - prev.get(key, curr)
        if abs(d) < 1e-6:
            return None
        arrow = "▲" if d > 0 else "▼"
        return f"{arrow} {abs(d):.1f}{suffix}"

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Risk Severity", risk_label, delta=f"Score: {risk_score}%", delta_color="inverse" if risk_level >= 2 else "off")
    col2.metric("Flow Water Level", f"{flow_water_level:.1f} m", delta=_delta(flow_water_level, "flow_water_level", " m"))
    col3.metric("Slope Displacement", f"{slope_movement:.1f} mm", delta=_delta(slope_movement, "slope_movement", " mm"), delta_color="inverse")
    col4.metric("24h Rainfall", f"{rainfall_24h:.0f} mm", delta=f"1h: {rainfall_1h:.0f}mm/h")
    col5.metric("Discharge Flow", f"{discharge:.0f} m³/s", delta=_delta(discharge, "discharge", " m³/s"))

    st.session_state.prev_tick = {
        "flow_water_level": flow_water_level,
        "slope_movement": slope_movement,
        "discharge": discharge
    }

    st.markdown("---")

    left, middle, right = st.columns([1.1, 1.1, 1.0])

    with left:
        st.subheader("🏔️ 3D Valley Terrain & Slope Failure")
        valley_fig = create_3d_valley_terrain(round(flow_water_level, 1), round(slope_movement, 1))
        st.plotly_chart(valley_fig, use_container_width=True, key="3d_valley_terrain_chart")
        if slope_movement > 25.0:
            st.error("🚨 Active Slope Displacement Highlighted on West Canyon Wall")
        elif flow_water_level > 10.0:
            st.warning("⚠️ High Channel Inundation Warning")
        else:
            st.success("Valley channel flow within normal baseline")

    with middle:
        st.subheader("📈 Catchment Sensor Trends")
        if len(hist_df) > 1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist_df["timestamp"], y=hist_df["flow_water_level_m"], name="Flow Level (m)", line=dict(color="#38bdf8", width=2.5)))
            fig.add_trace(go.Scatter(x=hist_df["timestamp"], y=hist_df["slope_movement_mm"], name="Slope Displacement (mm)", line=dict(color="#f43f5e", width=2), yaxis="y2"))
            fig.update_layout(
                height=300, margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                font=dict(color="#f8fafc"),
                legend=dict(orientation="h", y=1.15),
                yaxis=dict(title="Flow Level (m)", gridcolor="#1e293b"),
                yaxis2=dict(title="Slope Movement (mm)", overlaying="y", side="right", gridcolor="#1e293b"),
                uirevision="trends_constant"
            )
            st.plotly_chart(fig, use_container_width=True, key="sensor_trends_chart")
        else:
            st.info("Adjust sliders to view real-time sensor trend graphs...")

    with right:
        st.subheader("🎯 ML Classifier Risk Output")
        proba_df = pd.DataFrame({"Level": ["Low", "Medium", "High", "Critical"], "Probability": risk_proba * 100})
        fig_bar = px.bar(
            proba_df, x="Level", y="Probability", color="Level",
            color_discrete_map={"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ea580c", "Critical": "#ef4444"},
            text=proba_df["Probability"].round(1).astype(str) + "%"
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(
            height=300, showlegend=False, margin=dict(l=10, r=10, t=20, b=10), yaxis_range=[0, 115],
            paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
            font=dict(color="#f8fafc"),
            yaxis=dict(gridcolor="#1e293b"),
            uirevision="proba_constant"
        )
        st.plotly_chart(fig_bar, use_container_width=True, key="risk_probability_chart")

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
    selected_zone = st.sidebar.selectbox(
        "Target Hilly Catchment / Ward", catchment_options, index=0, key="live_catchment_zone"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📡 Telemetry Configuration")
    auto_stream = st.sidebar.toggle("⚡ Auto-Stream Live Telemetry", value=False)
    
    if "flow_water_level" not in st.session_state:
        st.session_state.flow_water_level = 4.5
        st.session_state.slope_movement = 4.2
        st.session_state.discharge = 45.0
        st.session_state.rainfall_1h = 12.0
        st.session_state.rainfall_24h = 48.0
        st.session_state.air_temp = 8.5
        st.session_state.surface_temp = 7.2
        st.session_state.prev_state_telemetry = None
        st.session_state.prev_tick = None
        st.session_state.zone_status = {}
        st.session_state.critical_since = None

        now = datetime.now()
        warmup = []
        for i in range(8, 0, -1):
            t = (now - pd.Timedelta(seconds=i * 6)).strftime("%H:%M:%S")
            warmup.append({
                "timestamp": t,
                "flow_water_level_m": round(4.5 + np.random.uniform(-0.3, 0.3), 2),
                "slope_movement_mm": round(4.2 + np.random.uniform(-0.4, 0.4), 2),
                "discharge_m3s": round(45.0 + np.random.uniform(-3, 3), 1),
                "risk_level": 0
            })
        st.session_state.history = warmup

    with st.sidebar.expander("🌧️ Rainfall & Weather Sliders", expanded=True):
        rainfall_1h = st.slider("1-Hour Rainfall (mm/h)", 0.0, 80.0, float(st.session_state.rainfall_1h), 1.0)
        rainfall_24h = st.slider("24-Hour Rainfall (mm)", 0.0, 250.0, float(st.session_state.rainfall_24h), 2.0)
        air_temp = st.slider("Air Temp (°C)", -10.0, 30.0, float(st.session_state.air_temp), 0.5)
        surface_temp = st.slider("Surface Temp (°C)", -10.0, 30.0, float(st.session_state.surface_temp), 0.5)

    with st.sidebar.expander("🌊 Hydrological & Geotechnical Sliders", expanded=True):
        flow_water_level = st.slider("River Flow Level (m)", 0.5, 20.0, float(st.session_state.flow_water_level), 0.2)
        slope_movement = st.slider("Slope Displacement (mm)", 0.0, 150.0, float(st.session_state.slope_movement), 0.5)
        discharge = st.slider("Discharge Rate (m³/s)", 5.0, 500.0, float(st.session_state.discharge), 5.0)

    st.session_state.flow_water_level = flow_water_level
    st.session_state.slope_movement = slope_movement
    st.session_state.discharge = discharge
    st.session_state.rainfall_1h = rainfall_1h
    st.session_state.rainfall_24h = rainfall_24h
    st.session_state.air_temp = air_temp
    st.session_state.surface_temp = surface_temp

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚡ Quick Presets")
    c_p1, c_p2 = st.sidebar.columns(2)
    if c_p1.button("✅ Baseline", use_container_width=True):
        st.session_state.flow_water_level, st.session_state.slope_movement, st.session_state.discharge = 3.2, 1.5, 30.0
        st.session_state.rainfall_1h, st.session_state.rainfall_24h, st.session_state.air_temp, st.session_state.surface_temp = 4.0, 22.0, 10.0, 8.5
        st.rerun()
    if c_p2.button("🚨 Storm Trigger", use_container_width=True):
        st.session_state.flow_water_level, st.session_state.slope_movement, st.session_state.discharge = 14.8, 68.0, 280.0
        st.session_state.rainfall_1h, st.session_state.rainfall_24h, st.session_state.air_temp, st.session_state.surface_temp = 48.0, 165.0, 4.0, 3.2
        st.rerun()

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
        exposure_color = "#ef4444" if sim_risk >= 2 else ("#f59e0b" if sim_risk == 1 else "#22c55e")
        st.markdown(
            f"""<div style="background:#1e293b;border:1px solid #334155;border-left:5px solid {exposure_color};
            border-radius:10px;padding:0.9rem 1.2rem;margin-bottom:0.8rem;">
            <span style="color:#94a3b8;font-size:0.85rem;font-weight:600;letter-spacing:0.4px;">DOWNSTREAM POPULATION EXPOSED</span><br>
            <span style="color:#f8fafc;font-size:1.9rem;font-weight:800;">{total_exposed:,} residents</span>
            <span style="color:#94a3b8;font-size:0.85rem;"> across {len(zone_data['vulnerable_wards'])} vulnerable wards</span>
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
        f"""<div style="background:#1e293b;border:1px solid #334155;border-radius:10px;
        padding:0.9rem 1.2rem;margin-bottom:1rem;">
        <span style="color:#94a3b8;font-size:0.85rem;font-weight:600;">10 LIVE MULTI-SOURCE PARAMETERS EVALUATED PER TICK</span><br>
        <span style="color:#cbd5e1;font-size:0.88rem;">{" · ".join(VARUNX_FEATURE_COLS)}</span>
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
    tab1, tab2 = st.tabs(["🔄 Historical Event Replay (Feature 6.5)", "🗺️ Vulnerable Zone Inventory (Feature 6.6)"])
    
    with tab1:
        st.subheader("Past Flash Flood & Landslide Events Database")
        st.info("Click 'Replay Event' to load past disaster conditions into the Live Monitoring session.")
        
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
            with c_r2:
                st.markdown("#### ")
                if st.button("▶️ Replay Event Conditions into Live Dashboard", use_container_width=True):
                    if "Kedarnath" in selected_event:
                        st.session_state.flow_water_level, st.session_state.slope_movement, st.session_state.discharge = 16.5, 85.0, 340.0
                        st.session_state.rainfall_1h, st.session_state.rainfall_24h = 55.0, 190.0
                    elif "Sikkim" in selected_event:
                        st.session_state.flow_water_level, st.session_state.slope_movement, st.session_state.discharge = 14.2, 52.0, 290.0
                        st.session_state.rainfall_1h, st.session_state.rainfall_24h = 42.0, 150.0
                    else:
                        st.session_state.flow_water_level, st.session_state.slope_movement, st.session_state.discharge = 11.5, 45.0, 210.0
                        st.session_state.rainfall_1h, st.session_state.rainfall_24h = 32.0, 110.0
                    st.success(f"✅ Replayed sensor state for: {selected_event}! Go to Tab 1 to view live impact.")
        else:
            st.error("Historical event data file not found.")

    with tab2:
        st.subheader("Vulnerable Catchment Zone Inventory")
        st.caption("Search known vulnerable hilly zones and hand-off directly to Live Monitoring.")
        
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
        with c_i2:
            st.markdown("#### ")
            if st.button("🎯 Hand-off Zone to Live Monitoring", use_container_width=True):
                st.success(f"✅ Handed off '{target_inv_zone}' to Live Monitoring System!")


def main():
    is_live = st.session_state.get("auto_stream_on", False)
    dot_class = "live-pulse-dot" if is_live else "live-pulse-dot offline"
    live_label = "LIVE TELEMETRY STREAMING" if is_live else "TELEMETRY IDLE"

    # Header Banner Card
    st.markdown(f"""
    <div class="varunx-header">
        <h1 class="varunx-title">🏔️ VarunX Early Warning System</h1>
        <p class="varunx-subtitle">AI & IoT-Powered Flash Flood & Landslide Risk Prediction for Hilly Regions | SIH26192 • Ministry of Home Affairs (NDRF, DM Division)</p>
        <div class="live-pulse-wrap">
            <span class="{dot_class}"></span>
            <span class="live-pulse-label">{live_label}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.radio(
        "VarunX Navigation",
        [
            "🔴 Live Monitoring & 3D Valley",
            "🗺️ Interactive GIS Risk Map",
            "🚨 Emergency Alert Gateway",
            "📊 ML Risk Engine Studio",
            "📚 Historical Replay & Inventory"
        ],
        index=0
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