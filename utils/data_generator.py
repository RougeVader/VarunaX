"""
Synthetic Data Generator for VarunX - Flash Flood & Landslide Early Warning System.
SIH26192 - Ministry of Home Affairs (NDRF, DM Division)
Creates realistic historical sensor + catchment weather data for model training.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

def generate_historical_data(n_samples=2500, seed=42):
    """
    Generate synthetic historical data matching VarunX PRD requirements:
    - rainfall_1h_mm, rainfall_3h_mm, rainfall_24h_mm
    - flow_water_level_m, water_level_rate_m_h
    - slope_movement_mm, tilt_rate_mm_h
    - discharge_m3s
    - air_temp_c, surface_temp_c
    """
    np.random.seed(seed)
    data = []
    base_time = datetime(2023, 1, 1)
    
    for i in range(n_samples):
        # Risk levels: 0=Low, 1=Medium, 2=High, 3=Critical
        risk_level = np.random.choice([0, 1, 2, 3], p=[0.45, 0.30, 0.18, 0.07])
        
        if risk_level == 0:  # Low risk - stable baseline
            flow_level = np.random.normal(3.5, 0.8)
            slope_movement = np.random.normal(1.2, 0.5)
            discharge = np.random.normal(30, 8)
            rainfall_1h = np.random.exponential(3)
            rainfall_3h = rainfall_1h * np.random.uniform(1.8, 2.5)
            rainfall_24h = rainfall_3h * np.random.uniform(2.0, 3.5)
            air_temp = np.random.normal(12, 4)
            surface_temp = air_temp - np.random.uniform(0.5, 2.0)
            
        elif risk_level == 1:  # Medium risk
            flow_level = np.random.normal(6.5, 1.2)
            slope_movement = np.random.normal(6.0, 2.0)
            discharge = np.random.normal(75, 15)
            rainfall_1h = np.random.exponential(10)
            rainfall_3h = rainfall_1h * np.random.uniform(2.0, 2.8)
            rainfall_24h = rainfall_3h * np.random.uniform(2.5, 4.0)
            air_temp = np.random.normal(10, 3)
            surface_temp = air_temp - np.random.uniform(0.5, 1.8)
            
        elif risk_level == 2:  # High risk
            flow_level = np.random.normal(11.0, 2.0)
            slope_movement = np.random.normal(25.0, 8.0)
            discharge = np.random.normal(180, 35)
            rainfall_1h = np.random.exponential(25)
            rainfall_3h = rainfall_1h * np.random.uniform(2.2, 3.0)
            rainfall_24h = rainfall_3h * np.random.uniform(2.8, 4.5)
            air_temp = np.random.normal(7, 3)
            surface_temp = air_temp - np.random.uniform(0.5, 1.5)
            
        else:  # Critical risk - imminent flash flood surge / hillside landslide
            flow_level = np.random.normal(16.5, 2.5)
            slope_movement = np.random.normal(75.0, 20.0)
            discharge = np.random.normal(380, 70)
            rainfall_1h = np.random.exponential(45)
            rainfall_3h = rainfall_1h * np.random.uniform(2.5, 3.5)
            rainfall_24h = rainfall_3h * np.random.uniform(3.0, 5.0)
            air_temp = np.random.normal(4, 2.5)
            surface_temp = air_temp - np.random.uniform(0.2, 1.2)
        
        flow_level = max(0.5, flow_level)
        slope_movement = max(0.0, slope_movement)
        discharge = max(2.0, discharge)
        rainfall_1h = max(0.0, rainfall_1h)
        rainfall_3h = max(rainfall_1h, rainfall_3h)
        rainfall_24h = max(rainfall_3h, rainfall_24h)
        
        water_level_rate = round(float(np.random.normal(0.05, 0.15) + (risk_level * 0.35)), 3)
        tilt_rate = round(slope_movement * float(np.random.uniform(0.05, 0.25)), 3)
        
        row = {
            "timestamp": base_time + timedelta(hours=i*6),
            "catchment_zone": np.random.choice([
                "Kedarnath_Valley", "Teesta_Basin", "Alaknanda_Chamoli",
                "Beas_Kullu_Valley", "Lahaul_Spiti"
            ]),
            "rainfall_1h_mm": round(rainfall_1h, 1),
            "rainfall_3h_mm": round(rainfall_3h, 1),
            "rainfall_24h_mm": round(rainfall_24h, 1),
            "flow_water_level_m": round(flow_level, 2),
            "water_level_rate_m_h": water_level_rate,
            "slope_movement_mm": round(slope_movement, 1),
            "tilt_rate_mm_h": tilt_rate,
            "discharge_m3s": round(discharge, 1),
            "air_temp_c": round(air_temp, 1),
            "surface_temp_c": round(surface_temp, 1),
            "risk_level": risk_level  # 0=Low, 1=Medium, 2=High, 3=Critical
        }
        data.append(row)
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    df = generate_historical_data(2500)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/historical_varunx_data.csv", index=False)
    print(f"Generated {len(df)} VarunX samples")
    print(df["risk_level"].value_counts().sort_index())
