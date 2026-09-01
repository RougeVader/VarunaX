"""
Schemas package for VarunX - Flash Flood & Landslide Early Warning System.
"""
from .sensor_data import SensorReading, RiskPrediction, AlertPayload

__all__ = ["SensorReading", "RiskPrediction", "AlertPayload"]
