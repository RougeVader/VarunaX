"""
VarunX GIS Mapping Engine - Flash Flood & Landslide Catchment Zones
SIH26192 - Ministry of Home Affairs (NDRF, DM Division)
Generates interactive spatial maps of Hilly Catchments,
risk markers, and downstream inundation/landslide threat vectors.
Compatible with Plotly v5, v6, and v7+.
"""

from typing import Dict, List, Optional
import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Hilly Catchment Registry with real scientific coordinates and downstream vulnerability points
CATCHMENT_ZONES_DATABASE = {
    "Kedarnath Valley (Mandakini Catchment)": {
        "lat": 30.7352, "lon": 79.0669, "elevation": 3584, "state": "Uttarakhand", "district": "Rudraprayag",
        "vulnerable_wards": [
            {"name": "Kedarnath Sanctuary Reach", "lat": 30.7450, "lon": 79.0600, "distance_km": 2, "type": "Temple & Pilgrim Hub", "population": 2500},
            {"name": "Rambara Valleyside Ward", "lat": 30.7000, "lon": 79.0500, "distance_km": 7, "type": "Debris Flow Corridor", "population": 1200},
            {"name": "Sonprayag Confluence Town", "lat": 30.6333, "lon": 79.0167, "distance_km": 18, "type": "River Confluence Settlement", "population": 4500}
        ]
    },
    "Teesta River Basin (North Sikkim)": {
        "lat": 27.6000, "lon": 88.6333, "elevation": 1550, "state": "Sikkim", "district": "Mangan",
        "vulnerable_wards": [
            {"name": "Chungthang Hydro Dam Ward", "lat": 27.6100, "lon": 88.6400, "distance_km": 3, "type": "Hydro Infrastructure & Dam", "population": 3100},
            {"name": "Dikchu Highway Bridge Ward", "lat": 27.3833, "lon": 88.5167, "distance_km": 28, "type": "Bridge & Highway Ward", "population": 2800},
            {"name": "Singtam Floodplain Settlement", "lat": 27.2333, "lon": 88.5000, "distance_km": 45, "type": "Urban Ward", "population": 8900}
        ]
    },
    "Solukhumbu Hilly Catchment (Everest Region)": {
        "lat": 27.8900, "lon": 86.8333, "elevation": 4350, "state": "Solukhumbu", "district": "High Himalaya",
        "vulnerable_wards": [
            {"name": "Dingboche Village Ward", "lat": 27.8800, "lon": 86.8200, "distance_km": 4, "type": "Hilly Settlement", "population": 1500},
            {"name": "Phakding River Corridor", "lat": 27.7500, "lon": 86.7167, "distance_km": 22, "type": "River Corridor Settlement", "population": 2100},
            {"name": "Lukla Valley Ward", "lat": 27.6869, "lon": 86.7314, "distance_km": 30, "type": "Transportation Hub", "population": 3800}
        ]
    },
    "Rolwaling Valley Catchment": {
        "lat": 27.8333, "lon": 86.4333, "elevation": 4180, "state": "Dolakha", "district": "Rolwaling",
        "vulnerable_wards": [
            {"name": "Na Village Ward", "lat": 27.8200, "lon": 86.4200, "distance_km": 3, "type": "High Slope Village", "population": 800},
            {"name": "Beding Slope Settlement", "lat": 27.8000, "lon": 86.3833, "distance_km": 10, "type": "Landslide Hazard Ward", "population": 1400},
            {"name": "Khimti Hydropower Station", "lat": 27.6167, "lon": 86.1333, "distance_km": 35, "type": "Power Grid", "population": 2200}
        ]
    },
    "Lahaul Valley (Chandra-Bhaga River)": {
        "lat": 32.4667, "lon": 77.1167, "elevation": 3120, "state": "Himachal Pradesh", "district": "Lahaul & Spiti",
        "vulnerable_wards": [
            {"name": "Sissu Catchment Ward", "lat": 32.4600, "lon": 77.1100, "distance_km": 2, "type": "Hilly Agricultural Ward", "population": 1900},
            {"name": "Manali-Leh Highway Corridor", "lat": 32.4500, "lon": 77.1000, "distance_km": 6, "type": "Strategic Highway", "population": 1100}
        ]
    }
}


def get_risk_color(risk_level: int) -> str:
    colors = {
        0: "#28a745",  # Green (Low)
        1: "#ffc107",  # Yellow (Medium)
        2: "#fd7e14",  # Orange (High)
        3: "#dc3545"   # Red (Critical)
    }
    return colors.get(risk_level, "#0d6efd")


def create_plotly_gis_map(selected_zone: str, current_risk_level: int, water_level: float, discharge: float):
    """
    Generate an interactive Plotly Map for VarunX showing Hilly Catchment Zones,
    highlighting the active zone with live risk markers and downstream vulnerability points.
    Compatible across Plotly 5.x, 6.x, and 7.x versions.
    """
    records = []
    
    for name, info in CATCHMENT_ZONES_DATABASE.items():
        is_selected = (name == selected_zone)
        r_level = current_risk_level if is_selected else 0
        r_label = ["LOW", "MEDIUM", "HIGH", "CRITICAL"][r_level]
        
        records.append({
            "name": name,
            "lat": info["lat"],
            "lon": info["lon"],
            "elevation": f"{info['elevation']} m",
            "state": info["state"],
            "risk_level": r_level,
            "risk_label": r_label,
            "marker_size": 26 if is_selected else 16,
            "selected": "Monitored Zone" if is_selected else "Catchment Area"
        })
    
    df = pd.DataFrame(records)
    
    if not HAS_PLOTLY:
        return df

    color_map = {
        "LOW": "#28a745",
        "MEDIUM": "#ffc107",
        "HIGH": "#fd7e14",
        "CRITICAL": "#dc3545"
    }

    # Plotly 7.0+ uses px.scatter_map instead of px.scatter_mapbox
    if hasattr(px, "scatter_map"):
        fig = px.scatter_map(
            df,
            lat="lat",
            lon="lon",
            color="risk_label",
            size="marker_size",
            hover_name="name",
            hover_data={"elevation": True, "state": True, "risk_label": True, "lat": False, "lon": False, "marker_size": False},
            color_discrete_map=color_map,
            zoom=5.8,
            center={"lat": 28.5, "lon": 83.5},
            title="<b>🗺️ VarunX: Hilly Region Flash Flood & Landslide Catchment Monitoring (NDRF)</b>"
        )
    elif hasattr(px, "scatter_mapbox"):
        fig = px.scatter_mapbox(
            df,
            lat="lat",
            lon="lon",
            color="risk_label",
            size="marker_size",
            hover_name="name",
            hover_data={"elevation": True, "state": True, "risk_label": True, "lat": False, "lon": False, "marker_size": False},
            color_discrete_map=color_map,
            zoom=5.8,
            center={"lat": 28.5, "lon": 83.5},
            title="<b>🗺️ VarunX: Hilly Region Flash Flood & Landslide Catchment Monitoring (NDRF)</b>"
        )
    else:
        # Fallback to px.scatter_geo
        fig = px.scatter_geo(
            df,
            lat="lat",
            lon="lon",
            color="risk_label",
            size="marker_size",
            hover_name="name",
            color_discrete_map=color_map,
            scope="asia",
            title="<b>🗺️ VarunX: Hilly Region Flash Flood & Landslide Catchment Monitoring (NDRF)</b>"
        )
    
    selected_info = CATCHMENT_ZONES_DATABASE.get(selected_zone)
    if selected_info:
        zone_lat = selected_info["lat"]
        zone_lon = selected_info["lon"]
        
        target_lats = [target["lat"] for target in selected_info["vulnerable_wards"]]
        target_lons = [target["lon"] for target in selected_info["vulnerable_wards"]]
        target_names = [f"⚠️ {target['name']} ({target['distance_km']}km, Pop: {target['population']})" for target in selected_info["vulnerable_wards"]]
        
        # Use scattergeo / scatter map traces depending on map style
        trace_scatter = go.Scattermap if hasattr(go, "Scattermap") else (go.Scattermapbox if hasattr(go, "Scattermapbox") else go.Scattergeo)
        
        fig.add_trace(trace_scatter(
            mode="markers+text",
            lat=target_lats,
            lon=target_lons,
            marker=dict(size=13, color="purple"),
            text=target_names,
            textposition="bottom right",
            name="Vulnerable Downstream Wards"
        ))
        
        for target in selected_info["vulnerable_wards"]:
            fig.add_trace(trace_scatter(
                mode="lines",
                lat=[zone_lat, target["lat"]],
                lon=[zone_lon, target["lon"]],
                line=dict(width=3, color="red" if current_risk_level >= 2 else "orange"),
                name=f"Flood & Debris Flow Path: {target['name']}"
            ))
    
    fig.update_layout(
        height=480,
        margin=dict(l=0, r=0, t=35, b=0),
        legend=dict(orientation="h", y=1.02, x=0.1)
    )
    
    return fig
