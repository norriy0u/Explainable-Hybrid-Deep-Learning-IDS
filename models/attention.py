import sys
from pathlib import Path
_base_dir = Path(__file__).resolve().parent
while 'config' not in [p.name for p in _base_dir.iterdir() if p.is_dir()] and _base_dir.parent != _base_dir:
    _base_dir = _base_dir.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

"""
Trainable Temporal Attention Mechanism Layer for Sequence Network IDS.
Calculates attention weights across sequence time steps:
e_t = tanh(h_t * W + b) * v
alpha_t = exp(e_t) / sum_j exp(e_j)
Context C = sum_t alpha_t * h_t
"""

import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras import layers, ops
from typing import Tuple, Dict, Any

@keras.saving.register_keras_serializable(package="Custom", name="TemporalAttention")
class TemporalAttention(layers.Layer):
    """
    Temporal Attention Mechanism Layer.
    Computes a learned weighted combination of time-step hidden representations.
    Exposes attention weights for visualization and explainability.
    """
    def __init__(self, return_attention_weights: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.return_attention_weights = return_attention_weights

    def build(self, input_shape):
        feature_dim = input_shape[-1]
        self.W = self.add_weight(
            shape=(feature_dim, feature_dim),
            initializer="glorot_uniform",
            trainable=True,
            name="attention_weight_matrix"
        )
        self.b = self.add_weight(
            shape=(feature_dim,),
            initializer="zeros",
            trainable=True,
            name="attention_bias"
        )
        self.v = self.add_weight(
            shape=(feature_dim, 1),
            initializer="glorot_uniform",
            trainable=True,
            name="attention_vector"
        )
        super().build(input_shape)

    def call(self, inputs):
        # inputs shape: (Batch, Time_Steps, Feature_Dim)
        # 1. Project inputs: score = tanh(inputs @ W + b)
        projected = ops.tanh(ops.matmul(inputs, self.W) + self.b)
        
        # 2. Alignment scores: e = projected @ v -> (Batch, Time_Steps, 1)
        alignment_scores = ops.matmul(projected, self.v)
        
        # 3. Softmax normalization across time axis: alpha_t = softmax(e)
        attention_weights = ops.softmax(alignment_scores, axis=1) # (B, T, 1)
        
        # 4. Compute weighted context vector: C = sum_t (inputs * alpha_t)
        context_vector = ops.sum(inputs * attention_weights, axis=1) # (B, D)
        
        if self.return_attention_weights:
            # Squeeze to (Batch, Time_Steps) for convenient visualization
            squeezed_weights = ops.squeeze(attention_weights, axis=-1)
            return context_vector, squeezed_weights
        
        return context_vector

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"return_attention_weights": self.return_attention_weights})
        return config
