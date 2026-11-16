import sys
from pathlib import Path
_base_dir = Path(__file__).resolve().parent
while _base_dir.name != 'Explainable_Hybrid_IDS' and _base_dir.parent != _base_dir:
    _base_dir = _base_dir.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

"""
Defensive Security Alert Generation Module for Network Security Operations Center (SOC).
Classifies threat severity, generates structured security alerts, and recommends defensive countermeasures.
"""

import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from config.config import SEVERITY_LEVELS, SEVERITY_COLORS

# Defensive incident response playbooks for detected network attacks
DEFENSIVE_PLAYBOOKS = {
    "DoS": (
        "ACTIVE DOS ATTACK: Implement immediate rate-limiting and connection throttling on ingress interfaces. "
        "Engage upstream scrubbing filters and apply SYN-cookie protection to alleviate half-open connection saturation."
    ),
    "Exploits": (
        "KNOWN EXPLOIT ATTEMPT: Inspect deep packet payloads for signature matches and shellcode strings. "
        "Verify target host patch level and isolate affected application server to prevent privilege escalation."
    ),
    "Worms": (
        "SELF-PROPAGATING WORM DETECTED: Immediately quarantine affected host network segment. "
        "Enforce strict egress port blocking on non-standard ports to arrest lateral propagation across subnet."
    ),
    "Shellcode": (
        "REMOTE SHELLCODE INJECTION: Terminate associated TCP stream and freeze target process memory for forensics. "
        "Alert SOC incident response team for immediate host endpoint inspection."
    ),
    "Backdoor": (
        "UNAUTHORIZED ACCESS / PERSISTENT BACKDOOR: Revoke host active credentials and terminate Command-and-Control (C2) channel. "
        "Audit host filesystem integrity and network sockets."
    ),
    "Reconnaissance": (
        "RECONNAISSANCE / SCAN DETECTED: Source host is mapping network topology and probing port states. "
        "Add source IP to dynamic perimeter observation list; enforce tarpit or stealth dropping on dark ports."
    ),
    "Fuzzers": (
        "PROTOCOL FUZZING: Abnormal, malformed packet fields received. "
        "Verify application protocol parser stability and discard malformed frames."
    ),
    "Analysis": (
        "SECURITY PROBING / ANALYSIS ACTIVITY: Port scans or HTML probing detected. "
        "Monitor source flow activity for subsequent exploit stages."
    ),
    "Generic": (
        "GENERIC ATTACK SIGNATURE: Traffic patterns match generic cryptographic or protocol collision attacks. "
        "Correlate with host firewall logs."
    ),
    "Normal": (
        "BENIGN NETWORK TRAFFIC: Traffic patterns conform to expected baseline communication protocols. "
        "No mitigation action required."
    )
}

def generate_security_alert(
    predicted_class: str,
    confidence: float,
    flow_id: Optional[str] = None,
    top_features: Optional[list] = None,
    explanation_summary: Optional[str] = None
) -> Dict[str, Any]:
    """
    Constructs a defensive SOC incident alert record.
    """
    now = datetime.now()
    alert_uuid = str(uuid.uuid4())[:8].upper()
    alert_id = f"SEC-{now.strftime('%Y%m%d')}-{alert_uuid}"
    
    severity = SEVERITY_LEVELS.get(predicted_class, "Medium")
    is_intrusion = (predicted_class != "Normal")
    color = SEVERITY_COLORS.get(severity, "#6B7280")
    playbook = DEFENSIVE_PLAYBOOKS.get(predicted_class, "Monitor flow and verify anomalous traffic signatures.")

    return {
        "alert_id": alert_id,
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp_iso": now.isoformat(),
        "is_intrusion": is_intrusion,
        "predicted_class": predicted_class,
        "confidence": round(float(confidence), 4),
        "confidence_pct": f"{confidence * 100:.2f}%",
        "severity": severity,
        "severity_color": color,
        "flow_identifier": flow_id or f"FLOW-{alert_uuid}",
        "top_features": top_features or [],
        "explanation_summary": explanation_summary or "SHAP explanation pending or not requested.",
        "defensive_playbook": playbook
    }
