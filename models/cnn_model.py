import sys
from pathlib import Path
_base_dir = Path(__file__).resolve().parent
while _base_dir.name != 'Explainable_Hybrid_IDS' and _base_dir.parent != _base_dir:
    _base_dir = _base_dir.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

"""
1D-CNN Baseline Deep Learning Model for Network Intrusion Detection.
Extracts spatial and localized flow patterns across input sequences.
"""

import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras import layers

def build_cnn_model(
    input_shape: tuple = (5, 42),
    num_classes: int = 10,
    filters: int = 64,
    kernel_size: int = 3,
    dropout_rate: float = 0.25,
    dense_units: int = 64,
    learning_rate: float = 0.001
) -> keras.Model:
    """Builds and compiles a standalone 1D-CNN multiclass classification model."""
    inputs = keras.Input(shape=input_shape, name="network_sequence_input")
    
    # 1D-CNN Layer
    x = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        activation="relu",
        name="conv1d_extractor"
    )(inputs)
    x = layers.BatchNormalization(name="batch_norm_cnn")(x)
    x = layers.Dropout(dropout_rate, name="dropout_cnn")(x)
    
    # Second Conv1D for deeper feature extraction
    x = layers.Conv1D(
        filters=filters // 2,
        kernel_size=kernel_size,
        padding="same",
        activation="relu",
        name="conv1d_refine"
    )(x)
    x = layers.GlobalAveragePooling1D(name="global_avg_pooling")(x)
    
    # Classification Head
    x = layers.Dense(dense_units, activation="relu", name="dense_features")(x)
    x = layers.Dropout(dropout_rate, name="dropout_dense")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="10_class_softmax")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="CNN_Baseline_Model")
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
