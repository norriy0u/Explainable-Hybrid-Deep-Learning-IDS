import sys
from pathlib import Path
_base_dir = Path(__file__).resolve().parent
while 'config' not in [p.name for p in _base_dir.iterdir() if p.is_dir()] and _base_dir.parent != _base_dir:
    _base_dir = _base_dir.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

"""
Real-Time and Batch Network Intrusion Detection Engine.
Integrates preprocessing, sequence construction, hybrid model inference,
confidence estimation, SHAP explainability, and defensive alert generation.
"""

import os
os.environ["KERAS_BACKEND"] = "torch"
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List
import numpy as np
import pandas as pd
import keras

from utils.logger import get_logger
from config.config import (
    MODELS_DIR,
    PREPROCESSORS_DIR,
    LABEL_MAPPING,
    CLASS_NAMES,
    DEFAULT_SEQUENCE_LENGTH,
    DEMO_ATTACK_CLASS
)
from preprocessing.pipeline import PreprocessingPipeline
from detection.alert_generator import generate_security_alert
from detection.detection_history import DetectionHistoryManager
from explainability.shap_explainer import IDSShapExplainer

logger = get_logger("PredictAttack")

class AttackDetectionEngine:
    """
    High-Performance Inference and Explanation Engine.
    Executes live or batch network intrusion classification on the trained Hybrid CNN-BiLSTM-Attention IDS.
    """
    def __init__(
        self,
        model_path: Optional[Path] = None,
        pipeline_path: Optional[Path] = None,
        background_dataset: Optional[np.ndarray] = None
    ):
        self.model_path = model_path or (MODELS_DIR / "hybrid_ids_model.keras")
        self.pipeline_path = pipeline_path or (PREPROCESSORS_DIR / "preprocessing_pipeline.joblib")
        
        # Load Preprocessing Pipeline
        if not self.pipeline_path.exists():
            raise FileNotFoundError(
                f"Preprocessing pipeline artifact not found at: {self.pipeline_path}. "
                "Train the model first or place artifacts in the artifacts directory."
            )
        self.pipeline = PreprocessingPipeline.load(self.pipeline_path)
        
        # Load Trained Model
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained model artifact not found at: {self.model_path}. "
                "Train the model first or place the .keras model in artifacts/models/."
            )
        from models.attention import TemporalAttention
        logger.info(f"Loading trained hybrid model from: {self.model_path}...")
        self.model = keras.models.load_model(
            self.model_path,
            compile=False,
            custom_objects={"TemporalAttention": TemporalAttention}
        )
        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        logger.info("Hybrid IDS Model loaded successfully.")
        
        self.history_mgr = DetectionHistoryManager()
        self.explainer: Optional[IDSShapExplainer] = None
        self._init_explainer(background_dataset)

    def _init_explainer(self, background_dataset: Optional[np.ndarray] = None) -> None:
        """Initializes the background baseline and SHAP explainer."""
        try:
            if background_dataset is None:
                # Create a standardized normal zero-baseline background sequence
                num_features = len(self.pipeline.feature_columns)
                T = self.pipeline.sequence_length
                # 30 random normal flows centered at zero (mean scaled)
                background_dataset = np.zeros((30, T, num_features), dtype=np.float32)
                
            self.explainer = IDSShapExplainer(
                model=self.model,
                background_sequences=background_dataset,
                feature_names=self.pipeline.feature_columns,
                max_background=30
            )
        except Exception as e:
            logger.warning(f"Could not initialize SHAP explainer at startup: {e}. Can be initialized on-demand.")

    def predict_dataframe_sequence(
        self,
        df_records: pd.DataFrame,
        explain: bool = True,
        flow_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Processes a sequence of network observations (from 1 to N rows) and generates detection and explanation.
        
        Parameters:
            df_records: DataFrame containing raw network traffic records.
            explain: If True, executes SHAP game-theoretic attribution.
            flow_id: Optional identifier for logging.
        """
        start_t = time.time()
        
        # 1. Transform raw records into standardized 3D sequence
        X_seq = self.pipeline.transform_inference(df_records)
        
        # If multiple sequences are constructed from a multi-row upload, we analyze the final window
        eval_sequence = X_seq[-1:] # (1, T, D)
        
        # 2. Run Forward Pass through Hybrid Model
        raw_preds = self.model.predict(eval_sequence, verbose=0)
        if isinstance(raw_preds, list):
            # Model with attention output [predictions, attention_weights]
            probs_matrix = raw_preds[0]
            attn_weights = raw_preds[1][0]
        else:
            probs_matrix = raw_preds
            attn_weights = np.ones((self.pipeline.sequence_length,)) / self.pipeline.sequence_length

        probs = probs_matrix[0] # (10,)
        latency_ms = (time.time() - start_t) * 1000.0

        # 3. Identify Predicted Class
        predicted_class_id = int(np.argmax(probs))
        predicted_class_name = LABEL_MAPPING.get(predicted_class_id, "Unknown")
        confidence = float(probs[predicted_class_id])
        
        prob_distribution = {LABEL_MAPPING[i]: round(float(probs[i]), 4) for i in range(10)}

        # 4. Generate SHAP Explanation
        explanation = None
        if explain and self.explainer is not None:
            explanation = self.explainer.explain_sequence_prediction(
                eval_sequence,
                target_class=predicted_class_id
            )

        # 5. Generate Defensive Security Alert
        summary_text = explanation["summary_text"] if explanation else None
        top_feats = explanation["top_features"] if explanation else []
        alert = generate_security_alert(
            predicted_class=predicted_class_name,
            confidence=confidence,
            flow_id=flow_id,
            top_features=top_feats,
            explanation_summary=summary_text
        )

        # 6. Record in Detection History
        self.history_mgr.record_detection(alert)

        result = {
            "predicted_class_id": predicted_class_id,
            "predicted_class": predicted_class_name,
            "confidence": round(confidence, 4),
            "confidence_pct": f"{confidence * 100:.2f}%",
            "is_intrusion": (predicted_class_name != "Normal"),
            "probabilities": prob_distribution,
            "latency_ms": round(latency_ms, 2),
            "model_name": "Proposed Hybrid (1D-CNN + BiLSTM + Attention)",
            "sequence_length": self.pipeline.sequence_length,
            "attention_weights": [round(float(w), 4) for w in attn_weights],
            "security_alert": alert,
            "explanation": explanation
        }

        logger.info(
            f"Detection Result: {predicted_class_name} ({confidence*100:.1f}%) "
            f"in {latency_ms:.1f}ms - Intrusion={result['is_intrusion']}"
        )
        return result
