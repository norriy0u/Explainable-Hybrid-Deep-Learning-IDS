import sys
from pathlib import Path
_base_dir = Path(__file__).resolve().parent
while _base_dir.name != 'Explainable_Hybrid_IDS' and _base_dir.parent != _base_dir:
    _base_dir = _base_dir.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

"""
Proposed Explainable Hybrid Deep Learning Architecture for Network Intrusion Detection:
Input Sequence -> 1D-CNN -> BiLSTM -> Temporal Attention -> Dense -> 10-Class Softmax
"""

import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras import layers
from models.attention import TemporalAttention
from utils.logger import get_logger

logger = get_logger("HybridModel")

def build_hybrid_ids_model(
    input_shape: tuple = (5, 42),
    num_classes: int = 10,
    filters: int = 64,
    kernel_size: int = 3,
    lstm_units: int = 64,
    dropout_rate: float = 0.25,
    dense_units: int = 64,
    learning_rate: float = 0.001,
    return_attention: bool = False
) -> keras.Model:
    """
    Builds the proposed CNN-BiLSTM-Attention Hybrid Architecture for 10-class network intrusion detection.
    
    Architecture:
      1. Input: (Batch, Sequence_Length=5, Features=D)
      2. 1D-CNN: Extracts localized flow patterns and correlations across features.
      3. BiLSTM: Captures bidirectional temporal relationships across consecutive observations.
      4. Attention: Dynamically prioritizes the most informative observation time steps.
      5. Dense & Regularization: Higher-level representation refinement with dropout.
      6. 10-Class Softmax: Outputs probability distribution over all ten attack/normal categories.
    """
    inputs = keras.Input(shape=input_shape, name="network_traffic_sequence")
    
    # --- MODULE A: 1D-CNN ---
    x = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        activation="relu",
        name="conv1d_local_feature_extractor"
    )(inputs)
    x = layers.BatchNormalization(name="batch_norm_cnn")(x)
    x = layers.Dropout(dropout_rate, name="spatial_dropout_cnn")(x)
    
    # --- MODULE B: BiLSTM ---
    # return_sequences=True is mandatory for the subsequent temporal attention module
    bilstm_seq = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True, dropout=0.2),
        name="bilstm_temporal_context"
    )(x)
    
    # --- MODULE C: TEMPORAL ATTENTION MECHANISM ---
    if return_attention:
        context_vector, attn_weights = TemporalAttention(
            return_attention_weights=True,
            name="temporal_attention_mechanism"
        )(bilstm_seq)
    else:
        context_vector = TemporalAttention(
            return_attention_weights=False,
            name="temporal_attention_mechanism"
        )(bilstm_seq)
        
    # --- MODULE D: DENSE AND 10-CLASS SOFTMAX HEAD ---
    dense_out = layers.Dense(dense_units, activation="relu", name="dense_classifier")(context_vector)
    dense_out = layers.BatchNormalization(name="batch_norm_dense")(dense_out)
    dense_out = layers.Dropout(dropout_rate, name="dense_dropout")(dense_out)
    
    predictions = layers.Dense(
        num_classes,
        activation="softmax",
        name="10_class_prediction_probabilities"
    )(dense_out)
    
    if return_attention:
        model = keras.Model(
            inputs=inputs,
            outputs=[predictions, attn_weights],
            name="Explainable_Hybrid_IDS_Model_With_Attention"
        )
    else:
        model = keras.Model(
            inputs=inputs,
            outputs=predictions,
            name="Explainable_Hybrid_IDS_Model"
        )
        
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    logger.info(
        f"Built Proposed Hybrid Model (Filters={filters}, LSTM_Units={lstm_units}, "
        f"Dense={dense_units}, Input={input_shape}, Classes={num_classes})."
    )
    return model
