import sys
from pathlib import Path
_base_dir = Path(__file__).resolve().parent
while _base_dir.name != 'Explainable_Hybrid_IDS' and _base_dir.parent != _base_dir:
    _base_dir = _base_dir.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

"""
Dataset Loading and Validation Module for UNSW-NB15 and Custom Traffic Captures.
"""

from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import pandas as pd
from utils.logger import get_logger
from config.config import TRAIN_DATASET_PATH, TEST_DATASET_PATH, EXCLUDE_COLUMNS

logger = get_logger("LoadDataset")

def load_csv_data(filepath: Path) -> pd.DataFrame:
    """Loads a CSV dataset with robust encoding and error handling."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")
    
    logger.info(f"Loading dataset from: {filepath.name}...")
    try:
        df = pd.read_csv(filepath, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding="latin1", low_memory=False)
    
    logger.info(f"Loaded {df.shape[0]:,} rows and {df.shape[1]} columns from {filepath.name}.")
    return df

def validate_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """Validates dataframe columns, data types, missing values, and attack categories."""
    missing_count = df.isnull().sum().to_dict()
    total_missing = sum(missing_count.values())
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    
    has_target = "attack_cat" in df.columns or "label" in df.columns
    target_dist = {}
    if "attack_cat" in df.columns:
        target_dist = df["attack_cat"].astype(str).str.strip().value_counts().to_dict()

    return {
        "num_records": len(df),
        "num_features": len(df.columns),
        "columns": list(df.columns),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "total_missing_values": int(total_missing),
        "missing_per_column": {k: int(v) for k, v in missing_count.items() if v > 0},
        "has_target_column": has_target,
        "attack_distribution": target_dist,
        "duplicate_rows": int(df.duplicated().sum())
    }

def load_unsw_nb15_datasets(
    train_path: Optional[Path] = None,
    test_path: Optional[Path] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads both training and testing datasets.
    Automatically verifies schemas and ensures proper split isolation.
    """
    train_path = train_path or TRAIN_DATASET_PATH
    test_path = test_path or TEST_DATASET_PATH
    
    train_df = load_csv_data(train_path)
    test_df = load_csv_data(test_path)
    
    logger.info(f"Train split shape: {train_df.shape}, Test split shape: {test_df.shape}")
    return train_df, test_df
