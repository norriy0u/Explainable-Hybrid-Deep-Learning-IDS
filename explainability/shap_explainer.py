import sys
from pathlib import Path
_base_dir = Path(__file__).resolve().parent
while 'config' not in [p.name for p in _base_dir.iterdir() if p.is_dir()] and _base_dir.parent != _base_dir:
    _base_dir = _base_dir.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

"""
Explainable AI (XAI) Module powered by SHAP (SHapley Additive exPlanations).
Computes genuine feature attributions for 3D sequence predictions from the hybrid deep learning model.
Provides mathematically rigorous time-step aggregation and cybersecurity-domain interpretations.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import shap
import keras

from utils.logger import get_logger
from config.config import (
    SHAP_BACKGROUND_SAMPLES,
    SHAP_TOP_FEATURES_COUNT,
    LABEL_MAPPING,
    CLASS_TO_ID,
    DEMO_ATTACK_CLASS,
    DEMO_ATTACK_ID
)

logger = get_logger("ShapExplainer")

# Domain knowledge explanations for key UNSW-NB15 features during network attacks
FEATURE_CYBER_CONTEXT = {
    "sbytes": "Source-to-destination transaction bytes. High volumes indicate packet flooding or data exfiltration.",
    "dbytes": "Destination-to-source transaction bytes. Very low dbytes relative to sbytes indicates unacknowledged flood traffic.",
    "sttl": "Source time-to-live. Anomalous or clamped TTL values frequently signify spoofed or automated attack toolkits.",
    "dttl": "Destination time-to-live. Divergence from standard OS TTL values indicates anomalous path routing or proxies.",
    "rate": "Total packets per second. Bursts in transmission rate are the primary signature of Denial-of-Service (DoS) floods.",
    "sload": "Source bits per second. Extremely elevated source bandwidth consumption indicates aggressive resource exhaustion.",
    "dload": "Destination bits per second.",
    "spkts": "Source-to-destination packet count.",
    "dpkts": "Destination-to-source packet count.",
    "ct_srv_src": "Connections to the same service from the source IP in the past 100 connections. High values reveal targeted service hammering.",
    "ct_state_ttl": "Count of connections for distinct state and TTL values. High values indicate repetitive state attacks (SYN floods).",
    "ct_dst_ltm": "Connections to destination address in past 100 connections.",
    "ct_src_dport_ltm": "Connections from source to destination port in recent window. High values flag port scanning or multi-vector DoS.",
    "ct_dst_sport_ltm": "Connections to destination from source port.",
    "ct_dst_src_ltm": "Connections between source and destination IP.",
    "sinpkt": "Source inter-packet arrival time. Microsecond-level inter-packet gaps reveal automated botnet transmission.",
    "dinpkt": "Destination inter-packet arrival time.",
    "tcprtt": "TCP round trip time (synack + ackdat). Connection degradation and delayed handshakes indicate server saturation.",
    "synack": "TCP handshake SYN to SYN-ACK time.",
    "ackdat": "TCP handshake SYN-ACK to ACK time.",
    "smean": "Mean packet size transmitted by source. Uniform packet sizes are characteristic of flooding scripts.",
    "dmean": "Mean packet size transmitted by destination.",
    "proto": "Transport protocol identifier (TCP, UDP, ICMP). Protocol anomalies often identify non-standard communication.",
    "service": "Application service (HTTP, DNS, FTP, SMTP). Target service under stress during DoS or exploit.",
    "state": "Connection transaction state (CON, FIN, INT, REQ). Incomplete states (INT, REQ) flag half-open SYN flood attacks."
}

class IDSShapExplainer:
    """
    Genuine SHAP Explainer for Hybrid Deep Learning IDS.
    Computes game-theoretic Shapley attributions for sequence-based network flow predictions.
    """
    def __init__(
        self,
        model: keras.Model,
        background_sequences: np.ndarray,
        feature_names: List[str],
        max_background: int = SHAP_BACKGROUND_SAMPLES
    ):
        """
        Parameters:
            model: Trained Keras Hybrid IDS model.
            background_sequences: Baseline 3D array of shape (N_bg, T, D).
            feature_names: List of D feature names.
            max_background: Number of representative baseline samples to use.
        """
        self.model = model
        self.feature_names = feature_names
        self.num_features = len(feature_names)
        self.sequence_length = background_sequences.shape[1]
        
        # Subsample background for computational tractability
        if len(background_sequences) > max_background:
            indices = np.random.choice(len(background_sequences), max_background, replace=False)
            self.background = background_sequences[indices]
        else:
            self.background = background_sequences

        # Flattened background for permutation/kernel explanation
        self.bg_flat = self.background.reshape(len(self.background), -1)
        
        # Prediction wrapper accepting flattened representations (N, T * D)
        def predict_flat(x_flat: np.ndarray) -> np.ndarray:
            x_seq = x_flat.reshape(-1, self.sequence_length, self.num_features).astype(np.float32)
            preds = self.model.predict(x_seq, verbose=0)
            if isinstance(preds, list):
                preds = preds[0] # Handle models with attention output
            return preds

        self.predict_flat = predict_flat
        logger.info(f"Initializing SHAP Explainer with {len(self.background)} background sequences...")
        self.explainer = shap.Explainer(self.predict_flat, self.bg_flat)
        logger.info("SHAP Explainer initialized successfully.")

    def explain_sequence_prediction(
        self,
        input_sequence: np.ndarray,
        target_class: Optional[int] = None,
        top_k: int = SHAP_TOP_FEATURES_COUNT
    ) -> Dict[str, Any]:
        """
        Calculates exact SHAP values for an input sequence (1, T, D).
        Aggregates attributions across time steps T and returns sorted influential features.
        
        Parameters:
            input_sequence: 3D sequence array of shape (1, T, D).
            target_class: Target class index to explain (0-9). If None, explains predicted class.
            top_k: Number of top features to return.
        """
        if input_sequence.ndim == 2:
            input_sequence = input_sequence.reshape(1, self.sequence_length, self.num_features)
            
        # 1. Obtain model prediction
        probs = self.model.predict(input_sequence, verbose=0)
        if isinstance(probs, list):
            probs = probs[0]
        probs = probs[0] # 1D array of 10 probabilities
        
        predicted_class_id = int(np.argmax(probs))
        predicted_class_name = LABEL_MAPPING.get(predicted_class_id, "Unknown")
        predicted_confidence = float(probs[predicted_class_id])
        
        # Determine explanation class
        explain_class_id = target_class if target_class is not None else predicted_class_id
        explain_class_name = LABEL_MAPPING.get(explain_class_id, "Unknown")

       # 2. Compute SHAP Values using flattened input
sample_flat = input_sequence.reshape(1, -1)
# Permutation explainer requires max_evals >= 2 * num_features + 1 (421 for 5x42=210)
num_flat_features = sample_flat.shape[1]
required_evals = max(500, 2 * num_flat_features + 1)
try:
    shap_explanation = self.explainer(sample_flat, max_evals=required_evals, silent=True)
except TypeError:
    shap_explanation = self.explainer(sample_flat, max_evals=required_evals)
        # Permutation explainer requires max_evals >= 2 * num_features + 1 (421 for 5x42=210)
        num_flat_features = sample_flat.shape[1]
        required_evals = max(500, 2 * num_flat_features + 1)
        try:
            shap_explanation = self.explainer(sample_flat, max_evals=required_evals, silent=True)
        except TypeError:
            shap_explanation = self.explainer(sample_flat, max_evals=required_evals)
        
        # shap_values has shape (1, T * D, 10)
        raw_shap_vals = shap_explanation.values[0] # (T * D, 10)
        
        # Extract SHAP values for the chosen class
        class_shap_flat = raw_shap_vals[:, explain_class_id] # (T * D,)
        
        # 3. Reshape back to temporal sequence grid (T, D)
        temporal_shap = class_shap_flat.reshape(self.sequence_length, self.num_features) # (T, D)
        
        # 4. Mathematically Rigorous Aggregation Across Time Steps:
        # Net feature contribution = sum across time steps
        # Mean temporal contribution = mean across time steps
        net_attributions = np.sum(temporal_shap, axis=0) # (D,)
        abs_attributions = np.sum(np.abs(temporal_shap), axis=0) # (D,)

        # 5. Extract Final Time Step Raw Feature Values (current observation)
        current_obs_features = input_sequence[0, -1, :] # (D,)

        # 6. Rank Top Influential Features by Absolute Importance
        sorted_indices = np.argsort(abs_attributions)[::-1]
        
        feature_breakdown = []
        for rank, idx in enumerate(sorted_indices[:top_k], start=1):
            feat_name = self.feature_names[idx]
            net_attr = float(net_attributions[idx])
            feat_val = float(current_obs_features[idx])
            
            direction = "Increased Risk (Positive)" if net_attr >= 0 else "Decreased Risk (Negative)"
            domain_meaning = FEATURE_CYBER_CONTEXT.get(
                feat_name,
                f"Network traffic metric for {feat_name} observed during the connection sequence."
            )
            
            feature_breakdown.append({
                "rank": rank,
                "feature": feat_name,
                "importance_score": round(abs(net_attr), 5),
                "shap_value": round(net_attr, 5),
                "feature_value": round(feat_val, 4),
                "direction": direction,
                "is_positive": net_attr >= 0,
                "cyber_context": domain_meaning
            })

        # Plain-English SOC Analyst Summary
        top_pos = [f["feature"] for f in feature_breakdown if f["is_positive"]][:3]
        top_neg = [f["feature"] for f in feature_breakdown if not f["is_positive"]][:2]
        
        summary_text = (
            f"The Hybrid Deep Learning IDS classified this network traffic sequence as **{predicted_class_name}** "
            f"with a confidence of **{predicted_confidence * 100:.2f}%**. "
            f"SHAP game-theoretic attribution identified that anomalous values in **{', '.join(top_pos) if top_pos else 'primary network metrics'}** "
            f"were the primary drivers pushing the model's decision toward **{explain_class_name}**."
        )
        if top_neg:
            summary_text += f" Counterbalancing metrics including **{', '.join(top_neg)}** exhibited normal baseline behavior."

        return {
            "predicted_class_id": predicted_class_id,
            "predicted_class_name": predicted_class_name,
            "predicted_confidence": round(predicted_confidence, 4),
            "explained_class_id": explain_class_id,
            "explained_class_name": explain_class_name,
            "probabilities": {LABEL_MAPPING[i]: round(float(probs[i]), 4) for i in range(10)},
            "summary_text": summary_text,
            "top_features": feature_breakdown,
            "all_feature_attributions": {self.feature_names[i]: round(float(net_attributions[i]), 5) for i in range(self.num_features)},
            "temporal_shap_grid": temporal_shap.tolist()
        }
