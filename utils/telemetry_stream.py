"""
VarunX Real-Time Telemetry Stream & Live Weather API Integration
SIH26192 - Ministry of Home Affairs (NDRF, DM Division)
Pulls live meteorological forecasts (Open-Meteo API) and simulates continuous
high-frequency IoT sensor telemetry streams for hilly catchment zones.
"""

import numpy as np
import requests
import logging
from datetime import datetime
from typing import Dict, Tuple, Optional

from schemas.sensor_data import SensorReading
from utils.gis_mapper import CATCHMENT_ZONES_DATABASE

logger = logging.getLogger("VarunX_TelemetryStream")


def fetch_live_weather(lat: float, lon: float) -> Dict[str, float]:
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,precipitation"
    try:
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            data = response.json()
            current = data.get("current", {})
            air_temp = float(current.get("temperature_2m", -1.5))
            precip = float(current.get("precipitation", 4.0))
            return {
                "air_temp_c": round(air_temp, 1),
                "surface_temp_c": round(air_temp - 1.2, 1),
                "rainfall_1h_mm": round(precip, 1),
                "rainfall_3h_mm": round(precip * 2.5, 1),
                "rainfall_24h_mm": round(precip * 8.0, 1),
                "live_status": "ONLINE (Open-Meteo API)"
            }
    except Exception as e:
        logger.warning(f"Live weather API fetch failed ({e}). Falling back to simulation mode.")
    
    return {
        "air_temp_c": 8.5,
        "surface_temp_c": 7.2,
        "rainfall_1h_mm": 12.0,
        "rainfall_3h_mm": 32.0,
        "rainfall_24h_mm": 78.0,
        "live_status": "OFFLINE (Simulated Fallback)"
    }


def generate_telemetry_tick(
    zone_name: str = "Kedarnath Valley (Mandakini Catchment)",
    previous_state: Optional[Dict] = None,
    force_critical: bool = False,
    use_live_weather: bool = True
) -> Tuple[SensorReading, Dict]:
    zone_info = CATCHMENT_ZONES_DATABASE.get(zone_name, CATCHMENT_ZONES_DATABASE["Kedarnath Valley (Mandakini Catchment)"])
    
    if previous_state is None:
        previous_state = {
            "flow_water_level_m": 4.5,
            "slope_movement_mm": 2.5,
            "discharge_m3s": 45.0,
            "rainfall_1h_mm": 8.0,
            "rainfall_24h_mm": 45.0,
            "air_temp_c": 8.5
        }
    
    weather_info = {"air_temp_c": 8.5, "surface_temp_c": 7.2, "rainfall_1h_mm": 8.0, "rainfall_3h_mm": 24.0, "rainfall_24h_mm": 55.0, "live_status": "Disabled"}
    if use_live_weather:
        weather_info = fetch_live_weather(zone_info["lat"], zone_info["lon"])
    
    water_level = previous_state["flow_water_level_m"] + float(np.random.normal(0.04, 0.15))
    slope_movement = previous_state["slope_movement_mm"] + float(np.random.normal(0.2, 0.8))
    discharge = previous_state["discharge_m3s"] + float(np.random.normal(1.2, 4.5))
    
    if force_critical:
        water_level += float(np.random.uniform(2.5, 5.0))
        slope_movement += float(np.random.uniform(15.0, 35.0))
        discharge += float(np.random.uniform(80.0, 150.0))
        weather_info["rainfall_1h_mm"] += float(np.random.uniform(25.0, 45.0))
        weather_info["rainfall_24h_mm"] += float(np.random.uniform(80.0, 140.0))
    
    water_level = max(1.0, min(25.0, water_level))
    slope_movement = max(0.0, min(250.0, slope_movement))
    discharge = max(5.0, min(650.0, discharge))
    
    water_level_rate = round(float(np.random.normal(0.08, 0.25) + (0.6 if force_critical else 0)), 3)
    tilt_rate = round(slope_movement * float(np.random.uniform(0.1, 0.4)), 3)
    
    sensor_model = SensorReading(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        catchment_zone=zone_name,
        flow_water_level_m=round(water_level, 2),
        water_level_rate_m_h=water_level_rate,
        slope_movement_mm=round(slope_movement, 1),
        tilt_rate_mm_h=tilt_rate,
        discharge_m3s=round(discharge, 1),
        rainfall_1h_mm=weather_info["rainfall_1h_mm"],
        rainfall_3h_mm=weather_info["rainfall_3h_mm"],
        rainfall_24h_mm=weather_info["rainfall_24h_mm"],
        air_temp_c=weather_info["air_temp_c"],
        surface_temp_c=weather_info["surface_temp_c"]
    )
    
    next_state = {
        "flow_water_level_m": water_level,
        "slope_movement_mm": slope_movement,
        "discharge_m3s": discharge,
        "rainfall_1h_mm": weather_info["rainfall_1h_mm"],
        "rainfall_24h_mm": weather_info["rainfall_24h_mm"],
        "weather_status": weather_info["live_status"]
    }
    
    return sensor_model, next_state
