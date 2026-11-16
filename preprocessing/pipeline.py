import sys
from pathlib import Path
_base_dir = Path(__file__).resolve().parent
while _base_dir.name != 'Explainable_Hybrid_IDS' and _base_dir.parent != _base_dir:
    _base_dir = _base_dir.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

"""
End-to-End Modular Preprocessing Pipeline.
Coordinates data cleaning, encoding, scaling, sequence construction, and artifact persistence.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import joblib
import numpy as np
import pandas as pd
from utils.logger import get_logger
from config.config import (
    PREPROCESSORS_DIR,
    EXCLUDE_COLUMNS,
    CATEGORICAL_FEATURES,
    LABEL_MAPPING,
    DEFAULT_SEQUENCE_LENGTH
)
from preprocessing.clean_data import clean_network_dataframe
from preprocessing.encode_features import CategoricalFeatureEncoder, TargetLabelEncoder
from preprocessing.feature_scaling import NumericalFeatureScaler
from preprocessing.sequence_builder import create_sequences, pad_or_replicate_for_single_inference

logger = get_logger("PreprocessingPipeline")

class PreprocessingPipeline:
    """
    Production Preprocessing Pipeline for the Explainable Hybrid IDS.
    Ensures exact reproducibility between training, testing, and live inference.
    """
    def __init__(self, sequence_length: int = DEFAULT_SEQUENCE_LENGTH):
        self.sequence_length = sequence_length
        self.cat_encoder = CategoricalFeatureEncoder(CATEGORICAL_FEATURES)
        self.label_encoder = TargetLabelEncoder()
        self.scaler = NumericalFeatureScaler()
        self.numerical_impute_values: Dict[str, float] = {}
        self.feature_columns: list[str] = []
        self.is_fitted = False

    def fit_transform_splits(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        stride: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Fits all preprocessing stages exclusively on the training partition,
        then transforms both train and test partitions, constructing sequences.
        Prevents any data leakage.
        """
        logger.info("Fitting preprocessing pipeline on training partition...")

        # 1. Clean training data and compute median imputation dictionary
        train_clean, clean_report, self.numerical_impute_values = clean_network_dataframe(
            train_df, is_training=True
        )

        # 2. Fit and transform categorical features on train
        train_encoded = self.cat_encoder.fit_transform(train_clean)

        # 3. Determine feature columns (exclude IDs and ground truth target columns)
        self.feature_columns = [
            c for c in train_encoded.columns if c not in EXCLUDE_COLUMNS
        ]
        logger.info(f"Identified {len(self.feature_columns)} input feature columns.")

        # 4. Fit and transform feature scaler on train features
        X_train_scaled = self.scaler.fit_transform(train_encoded, self.feature_columns)

        # 5. Encode target labels on train
        y_train_idx = self.label_encoder.transform(train_clean["attack_cat"])
        y_train_cat = self.label_encoder.to_categorical(y_train_idx)

        # 6. Transform test data using fitted preprocessors
        logger.info("Transforming testing partition using fitted pipeline...")
        test_clean, _, _ = clean_network_dataframe(
            test_df, numerical_impute_values=self.numerical_impute_values, is_training=False
        )
        test_encoded = self.cat_encoder.transform(test_clean)
        X_test_scaled = self.scaler.transform(test_encoded)
        y_test_idx = self.label_encoder.transform(test_clean["attack_cat"])
        y_test_cat = self.label_encoder.to_categorical(y_test_idx)

        # 7. Construct Sliding Window Sequences
        logger.info(f"Building training sequences (T={self.sequence_length}, stride={stride})...")
        X_train_seq, y_train_seq = create_sequences(
            X_train_scaled, y_train_cat, sequence_length=self.sequence_length, stride=stride
        )

        logger.info(f"Building testing sequences (T={self.sequence_length}, stride={stride})...")
        X_test_seq, y_test_seq = create_sequences(
            X_test_scaled, y_test_cat, sequence_length=self.sequence_length, stride=stride
        )

        self.is_fitted = True
        logger.info("Preprocessing and sequence construction complete.")
        return X_train_seq, y_train_seq, X_test_seq, y_test_seq

    def transform_inference(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforms raw network traffic (single observation or stream) for live detection.
        Outputs 3D sequence array of shape (N_seq, sequence_length, D).
        """
        if not self.is_fitted:
            raise RuntimeError("PreprocessingPipeline must be fitted or loaded before inference.")

        # Clean using saved median imputation
        df_clean, _, _ = clean_network_dataframe(
            df, numerical_impute_values=self.numerical_impute_values, is_training=False
        )
        
        # Categorical encoding
        df_encoded = self.cat_encoder.transform(df_clean)
        
        # Scale features
        X_scaled = self.scaler.transform(df_encoded)

        if len(X_scaled) < self.sequence_length:
            # Replicate or pad single observation to form valid sequence
            return pad_or_replicate_for_single_inference(X_scaled, self.sequence_length)
        else:
            X_seq, _ = create_sequences(X_scaled, sequence_length=self.sequence_length, stride=1)
            return X_seq

    def save(self, filepath: Optional[Path] = None) -> Path:
        """Serializes and saves pipeline artifacts to disk."""
        filepath = filepath or (PREPROCESSORS_DIR / "preprocessing_pipeline.joblib")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "sequence_length": self.sequence_length,
            "cat_encoder": self.cat_encoder.get_state(),
            "scaler": self.scaler.get_state(),
            "numerical_impute_values": self.numerical_impute_values,
            "feature_columns": self.feature_columns,
            "label_mapping": LABEL_MAPPING,
            "is_fitted": self.is_fitted
        }
        joblib.dump(state, filepath)
        logger.info(f"Saved preprocessing pipeline artifacts to: {filepath}")
        return filepath

    @classmethod
    def load(cls, filepath: Optional[Path] = None) -> "PreprocessingPipeline":
        """Loads and reconstructs fitted preprocessing pipeline from disk."""
        filepath = filepath or (PREPROCESSORS_DIR / "preprocessing_pipeline.joblib")
        if not filepath.exists():
            raise FileNotFoundError(f"Preprocessing artifact not found at: {filepath}")
            
        state = joblib.load(filepath)
        instance = cls(sequence_length=state.get("sequence_length", DEFAULT_SEQUENCE_LENGTH))
        instance.cat_encoder.set_state(state["cat_encoder"])
        instance.scaler.set_state(state["scaler"])
        instance.numerical_impute_values = state["numerical_impute_values"]
        instance.feature_columns = state["feature_columns"]
        instance.is_fitted = state["is_fitted"]
        logger.info(f"Loaded preprocessing pipeline from: {filepath}")
        return instance
