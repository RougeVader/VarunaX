"""
Pydantic Data Models for VarunX - Flash Flood & Landslide Early Warning System.
SIH26192 - Ministry of Home Affairs (NDRF, DM Division)
Provides strict type-checking, data validation, and serialization contracts.
"""

from typing import List, Dict, Optional
from datetime import datetime

try:
    from pydantic import BaseModel, Field
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    class BaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        def model_dump(self):
            return self.__dict__
    def Field(default=None, **kwargs):
        if callable(default):
            return default()
        return default


if HAS_PYDANTIC:
    class SensorReading(BaseModel):
        """Input payload validation for VarunX IoT sensor telemetry & weather feeds."""
        timestamp: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        catchment_zone: str = Field(default="Kedarnath_Valley", description="Hilly Catchment Zone or Ward Name")
        flow_water_level_m: float = Field(..., ge=0.0, description="Flow / River Water level in meters")
        water_level_rate_m_h: float = Field(default=0.0, description="Rate of water level change (m/h)")
        slope_movement_mm: float = Field(..., ge=0.0, description="Slope movement / ground tilt displacement (mm)")
        tilt_rate_mm_h: float = Field(default=0.0, description="Rate of slope displacement (mm/h)")
        discharge_m3s: float = Field(..., ge=0.0, description="Water discharge flow rate in m³/s")
        rainfall_1h_mm: float = Field(default=0.0, ge=0.0, description="1-hour rainfall intensity (mm)")
        rainfall_3h_mm: float = Field(default=0.0, ge=0.0, description="3-hour cumulative rainfall (mm)")
        rainfall_24h_mm: float = Field(default=0.0, ge=0.0, description="24-hour cumulative rainfall (mm)")
        air_temp_c: float = Field(..., description="Air temperature in Celsius")
        surface_temp_c: float = Field(..., description="Surface / Ground temperature in Celsius")

        class Config:
            populate_by_name = True


    class RiskPrediction(BaseModel):
        """Output schema for VarunX ML Risk Classifier Engine."""
        risk_level: int = Field(..., ge=0, le=3, description="Risk level (0=Low, 1=Medium, 2=High, 3=Critical)")
        risk_label: str = Field(..., description="Risk category string")
        probability_score: float = Field(..., ge=0.0, le=100.0, description="Confidence score %")
        probabilities: Dict[str, float] = Field(..., description="Per-class probability distribution")
        estimated_lead_time_hours: float = Field(..., ge=0.0, description="Evacuation lead time window in hours")
        primary_driver: str = Field(default="High 24h Cumulative Rainfall & Rapid Slope Displacement")


    class AlertPayload(BaseModel):
        """Schema for VarunX Emergency Alert Gateway dispatch (NDRF / DM Division)."""
        alert_id: str
        timestamp: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        catchment_zone: str
        risk_level: int
        risk_label: str
        lead_time_hours: float
        trigger_reason: str
        channels: List[str] = Field(default=["SMS", "Email", "Webhook", "NDMA_CAP_XML"])
        recipients: List[str] = Field(default=["NDRF Control Room (DM Division)", "State Disaster Authority (SDMA)", "District Collectorate", "Village Emergency Response Teams"])
        message_body: str
        status: str = Field(default="DISPATCHED")
else:
    class SensorReading(BaseModel):
        pass
    class RiskPrediction(BaseModel):
        pass
    class AlertPayload(BaseModel):
        pass
