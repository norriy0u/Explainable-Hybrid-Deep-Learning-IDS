import sys
from pathlib import Path
_base_dir = Path(__file__).resolve().parent
while 'config' not in [p.name for p in _base_dir.iterdir() if p.is_dir()] and _base_dir.parent != _base_dir:
    _base_dir = _base_dir.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

"""
Detection History Persistence and Auditing Module for Security Operations Center.
Maintains forensic records of past detections, supports querying, filtering, and CSV export.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from utils.logger import get_logger
from utils.helpers import load_json, save_json
from config.config import REPORTS_DIR

logger = get_logger("DetectionHistory")
HISTORY_FILE = REPORTS_DIR / "detection_history.json"

class DetectionHistoryManager:
    """Manages persistent logging and retrieval of network intrusion alerts."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or HISTORY_FILE
        self.history: List[Dict[str, Any]] = self._load()

    def _load(self) -> List[Dict[str, Any]]:
        """Loads historical records from disk."""
        data = load_json(self.storage_path)
        if isinstance(data, list):
            return data
        return []

    def record_detection(self, alert_record: Dict[str, Any]) -> None:
        """Appends a new alert to history and persists to disk."""
        self.history.insert(0, alert_record) # Most recent first
        # Keep maximum 500 records to prevent uncontrolled disk growth
        if len(self.history) > 500:
            self.history = self.history[:500]
        save_json(self.history, self.storage_path)
        logger.info(f"Recorded alert {alert_record['alert_id']} ({alert_record['predicted_class']}) to history.")

    def get_all(self) -> List[Dict[str, Any]]:
        return self.history

    def get_dataframe(self) -> pd.DataFrame:
        """Converts history records into a pandas DataFrame."""
        if not self.history:
            return pd.DataFrame(columns=[
                "alert_id", "timestamp", "predicted_class", "confidence",
                "severity", "flow_identifier", "is_intrusion"
            ])
        return pd.DataFrame(self.history)

    def filter_records(
        self,
        severity: Optional[str] = None,
        attack_category: Optional[str] = None,
        intrusions_only: bool = False
    ) -> List[Dict[str, Any]]:
        """Filters historical alerts by severity or attack category."""
        filtered = self.history
        if intrusions_only:
            filtered = [r for r in filtered if r.get("is_intrusion", False)]
        if severity and severity != "All":
            filtered = [r for r in filtered if r.get("severity") == severity]
        if attack_category and attack_category != "All":
            filtered = [r for r in filtered if r.get("predicted_class") == attack_category]
        return filtered

    def export_csv(self, output_path: Optional[Path] = None) -> Path:
        """Exports detection history to CSV."""
        output_path = output_path or (REPORTS_DIR / "detection_history_export.csv")
        df = self.get_dataframe()
        df.to_csv(output_path, index=False)
        logger.info(f"Exported detection history ({len(df)} records) to: {output_path}")
        return output_path

    def clear(self) -> None:
        """Clears detection history."""
        self.history = []
        save_json([], self.storage_path)
        logger.info("Cleared detection history.")
