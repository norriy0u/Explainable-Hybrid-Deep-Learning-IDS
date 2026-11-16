import sys
from pathlib import Path
_base_dir = Path(__file__).resolve().parent
while 'config' not in [p.name for p in _base_dir.iterdir() if p.is_dir()] and _base_dir.parent != _base_dir:
    _base_dir = _base_dir.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

"""
Unit Tests for Attack Detection Engine and Alert Generator.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from detection.alert_generator import generate_security_alert

def test_alert_generation():
    # Test DoS alert
    alert = generate_security_alert(
        predicted_class="DoS",
        confidence=0.985,
        flow_id="FLOW-TEST-001"
    )
    assert alert["predicted_class"] == "DoS"
    assert alert["severity"] == "Critical"
    assert alert["is_intrusion"] is True
    assert "rate-limiting" in alert["defensive_playbook"].lower()
    
    # Test Normal alert
    alert_normal = generate_security_alert(
        predicted_class="Normal",
        confidence=0.992
    )
    assert alert_normal["severity"] == "Informational"
    assert alert_normal["is_intrusion"] is False
    print("Alert generator tests passed successfully!")

if __name__ == "__main__":
    test_alert_generation()
    print("ALL PREDICTION TESTS PASSED!")
