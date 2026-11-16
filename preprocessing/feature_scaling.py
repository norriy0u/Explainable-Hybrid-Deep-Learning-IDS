import sys
from pathlib import Path
_base_dir = Path(__file__).resolve().parent
while 'config' not in [p.name for p in _base_dir.iterdir() if p.is_dir()] and _base_dir.parent != _base_dir:
    _base_dir = _base_dir.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

"""
Numerical Feature Scaling Module for Network Intrusion Detection.
Applies StandardScaler (fitted exclusively on training split) to eliminate data leakage.
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils.logger import get_logger

logger = get_logger("FeatureScaling")

class NumericalFeatureScaler:
    """
    Standardizes features by removing the mean and scaling to unit variance:
    z = (x - u) / s
    Fitted strictly on training data; stores mean, scale, and feature names.
    """
    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names or []
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, feature_columns: Optional[List[str]] = None) -> "NumericalFeatureScaler":
        """Fits StandardScaler on provided training features."""
        self.feature_names = feature_columns or list(df.columns)
        X = df[self.feature_names].values.astype(np.float32)
        
        # Replace any residual NaNs with column means before scaling
        col_means = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_means, inds[1])
        
        self.scaler.fit(X)
        self.is_fitted = True
        logger.info(f"StandardScaler successfully fitted on {len(self.feature_names)} features.")
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transforms input dataframe into scaled numpy array."""
        if not self.is_fitted:
            raise RuntimeError("NumericalFeatureScaler must be fitted before transforming.")
            
        # Ensure all expected features are present; fill missing with 0.0
        missing_features = [f for f in self.feature_names if f not in df.columns]
        if missing_features:
            logger.warning(f"Features missing during scaling: {missing_features}. Filling with 0.0.")
            df_aligned = df.copy()
            for mf in missing_features:
                df_aligned[mf] = 0.0
        else:
            df_aligned = df
            
        X = df_aligned[self.feature_names].values.astype(np.float32)
        
        # Handle any residual NaNs
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        X_scaled = self.scaler.transform(X)
        return X_scaled

    def fit_transform(self, df: pd.DataFrame, feature_columns: Optional[List[str]] = None) -> np.ndarray:
        return self.fit(df, feature_columns).transform(df)

    def get_state(self) -> Dict[str, Any]:
        """Returns serializable dictionary of scaler state."""
        return {
            "feature_names": self.feature_names,
            "mean": self.scaler.mean_.tolist() if self.is_fitted else [],
            "scale": self.scaler.scale_.tolist() if self.is_fitted else [],
            "var": self.scaler.var_.tolist() if self.is_fitted else [],
            "n_samples_seen": int(self.scaler.n_samples_seen_) if self.is_fitted else 0,
            "is_fitted": self.is_fitted
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restores scaler from serialized state."""
        self.feature_names = state["feature_names"]
        self.scaler.mean_ = np.array(state["mean"], dtype=np.float64)
        self.scaler.scale_ = np.array(state["scale"], dtype=np.float64)
        self.scaler.var_ = np.array(state["var"], dtype=np.float64)
        self.scaler.n_samples_seen_ = state["n_samples_seen"]
        self.is_fitted = state["is_fitted"]
