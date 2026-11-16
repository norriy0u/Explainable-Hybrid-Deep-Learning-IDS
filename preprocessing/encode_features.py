import sys
from pathlib import Path
_base_dir = Path(__file__).resolve().parent
while 'config' not in [p.name for p in _base_dir.iterdir() if p.is_dir()] and _base_dir.parent != _base_dir:
    _base_dir = _base_dir.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

"""
Feature and Target Encoding Module for Network Traffic.
Supports categorical feature mapping with unseen-category handling and 10-class target encoding.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from utils.logger import get_logger
from config.config import LABEL_MAPPING, CLASS_TO_ID, CATEGORICAL_FEATURES

logger = get_logger("EncodeFeatures")

class CategoricalFeatureEncoder:
    """
    Encodes categorical network features into integers.
    Fitted strictly on training data; unknown values in test/inference map to a dedicated unknown index.
    """
    def __init__(self, categorical_columns: Optional[List[str]] = None):
        self.categorical_columns = categorical_columns or CATEGORICAL_FEATURES
        self.category_maps: Dict[str, Dict[str, int]] = {}
        self.unknown_indices: Dict[str, int] = {}
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> "CategoricalFeatureEncoder":
        """Fits category mappings on training dataframe."""
        self.category_maps = {}
        self.unknown_indices = {}
        
        for col in self.categorical_columns:
            if col in df.columns:
                unique_vals = sorted(df[col].astype(str).unique().tolist())
                mapping = {val: idx for idx, val in enumerate(unique_vals)}
                # Unknown index is set to len(unique_vals)
                self.unknown_indices[col] = len(unique_vals)
                self.category_maps[col] = mapping
                logger.info(f"Fitted encoder for '{col}': {len(unique_vals)} categories.")
            else:
                logger.warning(f"Categorical column '{col}' not found in dataframe during fit.")
                
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms categorical columns using fitted mappings."""
        if not self.is_fitted:
            raise RuntimeError("CategoricalFeatureEncoder must be fitted before transforming.")
            
        df_out = df.copy()
        for col in self.categorical_columns:
            if col in df_out.columns:
                mapping = self.category_maps.get(col, {})
                unk_idx = self.unknown_indices.get(col, 0)
                # Apply mapping with fallback to unknown index
                df_out[col] = df_out[col].astype(str).map(mapping).fillna(unk_idx).astype(int)
        return df_out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def get_state(self) -> Dict[str, Any]:
        """Returns serializable dictionary of fitted encoder state."""
        return {
            "categorical_columns": self.categorical_columns,
            "category_maps": self.category_maps,
            "unknown_indices": self.unknown_indices,
            "is_fitted": self.is_fitted
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restores encoder from serialized state."""
        self.categorical_columns = state["categorical_columns"]
        self.category_maps = state["category_maps"]
        self.unknown_indices = state["unknown_indices"]
        self.is_fitted = state["is_fitted"]


class TargetLabelEncoder:
    """
    Encodes the 10-class network attack categories strictly according to project specification:
    0 = Normal, 1 = Analysis, 2 = Backdoor, 3 = DoS, 4 = Exploits,
    5 = Fuzzers, 6 = Generic, 7 = Reconnaissance, 8 = Shellcode, 9 = Worms
    """
    def __init__(self):
        self.label_mapping = LABEL_MAPPING
        self.class_to_id = CLASS_TO_ID
        self.num_classes = len(self.label_mapping)

    def transform(self, labels: pd.Series) -> np.ndarray:
        """Converts string attack categories into 1D integer array."""
        clean_labels = labels.astype(str).str.strip()
        encoded = clean_labels.map(self.class_to_id)
        
        # Check for unmapped labels
        unmapped = encoded.isnull()
        if unmapped.any():
            unmapped_vals = clean_labels[unmapped].unique().tolist()
            logger.warning(f"Unmapped attack categories detected: {unmapped_vals}. Defaulting to 'Normal' (0).")
            encoded = encoded.fillna(0)
            
        return encoded.astype(int).values

    def inverse_transform(self, class_indices: np.ndarray) -> List[str]:
        """Converts integer class indices back to canonical attack category names."""
        return [self.label_mapping.get(int(idx), "Unknown") for idx in class_indices]

    def to_categorical(self, class_indices: np.ndarray) -> np.ndarray:
        """Converts 1D integer class array into one-hot encoded matrix (N, 10)."""
        one_hot = np.zeros((len(class_indices), self.num_classes), dtype=np.float32)
        for i, val in enumerate(class_indices):
            if 0 <= val < self.num_classes:
                one_hot[i, val] = 1.0
            else:
                one_hot[i, 0] = 1.0  # fallback to Normal
        return one_hot
