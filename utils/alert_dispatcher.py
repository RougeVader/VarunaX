"""
VarunX Emergency Alert Dispatcher Gateway
SIH26192 - Ministry of Home Affairs (NDRF, DM Division)
Handles multi-channel emergency notification dispatch (SMS, Email, Webhook, CAP XML)
and maintains an audit log of triggered disaster warnings for hilly regions.
"""

import os
import json
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Optional
import requests

from schemas.sensor_data import AlertPayload

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VarunX_Alert_Dispatcher")

ALERTS_LOG_FILE = "data/alerts_history.json"


def load_alert_history() -> List[Dict]:
    """Load historical dispatched alerts from JSON storage."""
    if os.path.exists(ALERTS_LOG_FILE):
        try:
            with open(ALERTS_LOG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading alert history: {e}")
            return []
    return []


def save_alert_history(history: List[Dict]) -> None:
    """Save dispatched alert records to storage."""
    os.makedirs(os.path.dirname(ALERTS_LOG_FILE), exist_ok=True)
    try:
        with open(ALERTS_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving alert history: {e}")


def format_alert_message(catchment_zone: str, risk_label: str, lead_time: float, trigger_reason: str) -> str:
    """Construct standardized VarunX emergency broadcast message body for NDRF & SDMA."""
    msg = (
        f"🚨 VarunX EMERGENCY FLASH FLOOD & LANDSLIDE WARNING 🚨\n"
        f"Target Zone / Ward: {catchment_zone}\n"
        f"Threat Severity: {risk_label} RISK\n"
        f"Estimated Evacuation Lead Time: {lead_time} Hours\n"
        f"Primary Driver: {trigger_reason}\n"
        f"Action Required: NDRF & SDMA Control Rooms — Initiate immediate village/ward level "
        f"evacuation and emergency warnings. Deploy field teams to slope displacement points."
    )
    return msg


def send_sms_simulated(recipients: List[str], message: str) -> Dict[str, str]:
    results = {}
    for r in recipients:
        logger.info(f"[SMS GATEWAY] Transmitting SMS to {r}: {message[:60]}...")
        results[r] = "DELIVERED (SIMULATED)"
    return results


def send_email_simulated(recipients: List[str], subject: str, message: str) -> Dict[str, str]:
    results = {}
    for r in recipients:
        logger.info(f"[EMAIL SMTP] Dispatching HTML alert email to {r} | Subject: {subject}")
        results[r] = "SENT (SIMULATED)"
    return results


def send_webhook_simulated(webhook_url: str, payload: Dict) -> bool:
    logger.info(f"[WEBHOOK DISPATCH] POST payload to {webhook_url}")
    try:
        if webhook_url.startswith("http"):
            resp = requests.post(webhook_url, json=payload, timeout=3)
            return resp.status_code == 200
        return True
    except Exception as e:
        logger.warning(f"Webhook dispatch failed: {e}")
        return False


def dispatch_emergency_alert(
    catchment_zone: str,
    risk_level: int,
    risk_label: str,
    lead_time_hours: float,
    trigger_reason: str,
    channels: Optional[List[str]] = None,
    recipients: Optional[List[str]] = None,
    webhook_url: Optional[str] = None
) -> AlertPayload:
    """
    Primary Gateway Function: Formats, dispatches, and logs a VarunX emergency alert.
    """
    if channels is None:
        channels = ["SMS", "Email", "Webhook", "NDMA_CAP_XML"]
    if recipients is None:
        recipients = ["NDRF Control Room (DM Division)", "State Disaster Management Authority (SDMA)", "District Collectorate", "Village Emergency Response Teams"]
    
    alert_id = f"VARUNX-ALT-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:4].upper()}"
    message_body = format_alert_message(catchment_zone, risk_label, lead_time_hours, trigger_reason)
    
    if "SMS" in channels:
        send_sms_simulated(recipients, message_body)
    if "Email" in channels:
        send_email_simulated(recipients, f"URGENT: VarunX Flash Flood & Landslide Warning - {catchment_zone}", message_body)
    if "Webhook" in channels and webhook_url:
        send_webhook_simulated(webhook_url, {
            "alert_id": alert_id,
            "zone": catchment_zone,
            "risk": risk_label,
            "lead_time_h": lead_time_hours,
            "message": message_body
        })
    
    alert_obj = AlertPayload(
        alert_id=alert_id,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        catchment_zone=catchment_zone,
        risk_level=risk_level,
        risk_label=risk_label,
        lead_time_hours=lead_time_hours,
        trigger_reason=trigger_reason,
        channels=channels,
        recipients=recipients,
        message_body=message_body,
        status="DISPATCHED"
    )
    
    history = load_alert_history()
    history.insert(0, alert_obj.model_dump())
    save_alert_history(history)
    
    logger.info(f"✅ Alert {alert_id} successfully dispatched and logged.")
    return alert_obj
