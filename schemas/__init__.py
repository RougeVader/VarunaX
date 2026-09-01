"""
Schemas package for GLOF Early Warning System.
"""
from .sensor_data import SensorReading, RiskPrediction, AlertPayload

__all__ = ["SensorReading", "RiskPrediction", "AlertPayload"]
