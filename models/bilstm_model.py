import sys
from pathlib import Path
_base_dir = Path(__file__).resolve().parent
while _base_dir.name != 'Explainable_Hybrid_IDS' and _base_dir.parent != _base_dir:
    _base_dir = _base_dir.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

"""
Bidirectional LSTM Baseline Model for Network Intrusion Detection.
Captures forward and backward temporal dependencies across network traffic observations.
"""

import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras import layers

def build_bilstm_model(
    input_shape: tuple = (5, 42),
    num_classes: int = 10,
    lstm_units: int = 64,
    dropout_rate: float = 0.25,
    dense_units: int = 64,
    learning_rate: float = 0.001
) -> keras.Model:
    """Builds and compiles a standalone BiLSTM multiclass classification model."""
    inputs = keras.Input(shape=input_shape, name="network_sequence_input")
    
    # Bidirectional LSTM Layer
    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=False, dropout=0.2),
        name="bilstm_temporal_learner"
    )(inputs)
    x = layers.BatchNormalization(name="batch_norm_bilstm")(x)
    
    # Classification Head
    x = layers.Dense(dense_units, activation="relu", name="dense_features")(x)
    x = layers.Dropout(dropout_rate, name="dropout_dense")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="10_class_softmax")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="BiLSTM_Baseline_Model")
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
