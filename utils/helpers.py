import sys
from pathlib import Path
_base_dir = Path(__file__).resolve().parent
while 'config' not in [p.name for p in _base_dir.iterdir() if p.is_dir()] and _base_dir.parent != _base_dir:
    _base_dir = _base_dir.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

"""
Utility Helper Functions for File Operations, Metrics, and Serialization.
"""

import json
from pathlib import Path
from typing import Any, Dict
import numpy as np

def save_json(data: Dict[str, Any], filepath: Path) -> None:
    """Safely saves a dictionary to a JSON file, converting numpy types."""
    def default_converter(o):
        if isinstance(o, (np.integer, np.int64, np.int32)):
            return int(o)
        if isinstance(o, (np.floating, np.float64, np.float32)):
            return float(o)
        if isinstance(o, (np.ndarray, list)):
            return [default_converter(x) for x in o]
        return str(o)

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, default=default_converter)

def load_json(filepath: Path) -> Dict[str, Any]:
    """Loads a JSON file if it exists."""
    if not filepath.exists():
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def format_percentage(val: float) -> str:
    """Formats float as percentage string."""
    return f"{val * 100:.2f}%"
