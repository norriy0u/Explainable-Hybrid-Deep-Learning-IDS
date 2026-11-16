import sys
from pathlib import Path
_base_dir = Path(__file__).resolve().parent
while 'config' not in [p.name for p in _base_dir.iterdir() if p.is_dir()] and _base_dir.parent != _base_dir:
    _base_dir = _base_dir.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

"""
Centralized Configuration for Explainable Hybrid Deep Learning IDS.
Defines paths, feature schemas, class labels, and default hyperparameters.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
DATASET_DIR = BASE_DIR / "dataset"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
PREPROCESSORS_DIR = ARTIFACTS_DIR / "preprocessors"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
SHAP_PLOTS_DIR = RESULTS_DIR / "shap_plots"
REPORTS_DIR = RESULTS_DIR / "reports"
EXPERIMENT_RESULTS_DIR = RESULTS_DIR / "experiment_results"

# Ensure runtime directories exist
for p in [MODELS_DIR, PREPROCESSORS_DIR, PLOTS_DIR, SHAP_PLOTS_DIR, REPORTS_DIR, EXPERIMENT_RESULTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Default Dataset File Paths
TRAIN_DATASET_PATH = DATASET_DIR / "UNSW_NB15_training-set.csv"
TEST_DATASET_PATH = DATASET_DIR / "UNSW_NB15_testing-set.csv"
FEATURES_METADATA_PATH = DATASET_DIR / "NUSW-NB15_features.csv"

# Target Multiclass Mapping (Exact 10 classes)
LABEL_MAPPING = {
    0: "Normal",
    1: "Analysis",
    2: "Backdoor",
    3: "DoS",
    4: "Exploits",
    5: "Fuzzers",
    6: "Generic",
    7: "Reconnaissance",
    8: "Shellcode",
    9: "Worms"
}

CLASS_NAMES = [LABEL_MAPPING[i] for i in range(10)]
CLASS_TO_ID = {name: i for i, name in LABEL_MAPPING.items()}

# Primary Demonstration Attack
DEMO_ATTACK_CLASS = "DoS"
DEMO_ATTACK_ID = CLASS_TO_ID[DEMO_ATTACK_CLASS]

# Columns to Exclude from Model Input Features
EXCLUDE_COLUMNS = ["id", "attack_cat", "label"]

# Categorical and Numerical Features in UNSW-NB15
CATEGORICAL_FEATURES = ["proto", "service", "state"]

# Alert Severity Mapping
SEVERITY_LEVELS = {
    "Normal": "Informational",
    "Analysis": "Medium",
    "Backdoor": "Critical",
    "DoS": "Critical",
    "Exploits": "High",
    "Fuzzers": "Medium",
    "Generic": "Medium",
    "Reconnaissance": "Low",
    "Shellcode": "Critical",
    "Worms": "Critical"
}

SEVERITY_COLORS = {
    "Informational": "#10B981", # Emerald
    "Low": "#06B6D4",           # Cyan
    "Medium": "#F59E0B",        # Amber
    "High": "#F97316",          # Orange
    "Critical": "#EF4444"       # Crimson
}

# Default Model & Sequence Hyperparameters
DEFAULT_SEQUENCE_LENGTH = 5
DEFAULT_BATCH_SIZE = 128
DEFAULT_EPOCHS = 15
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_CNN_FILTERS = 64
DEFAULT_CNN_KERNEL_SIZE = 3
DEFAULT_LSTM_UNITS = 64
DEFAULT_DROPOUT_RATE = 0.25
DEFAULT_DENSE_UNITS = 64
DEFAULT_VALIDATION_SPLIT = 0.15

# SHAP Configuration
SHAP_BACKGROUND_SAMPLES = 50
SHAP_TOP_FEATURES_COUNT = 10
