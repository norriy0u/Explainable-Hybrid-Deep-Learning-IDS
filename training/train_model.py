import sys
from pathlib import Path
_base_dir = Path(__file__).resolve().parent
while _base_dir.name != 'Explainable_Hybrid_IDS' and _base_dir.parent != _base_dir:
    _base_dir = _base_dir.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

"""
End-to-End Model Training Pipeline.
Coordinates data preprocessing, class-weighted learning, early stopping,
checkpointing, evaluation, and artifact persistence.
"""

import os
os.environ["KERAS_BACKEND"] = "torch"
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import keras
from keras import callbacks

from utils.logger import get_logger
from utils.helpers import save_json
from config.config import (
    MODELS_DIR,
    REPORTS_DIR,
    PLOTS_DIR,
    DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_SEQUENCE_LENGTH,
    DEFAULT_CNN_FILTERS,
    DEFAULT_LSTM_UNITS,
    DEFAULT_DROPOUT_RATE,
    DEFAULT_DENSE_UNITS,
    DEFAULT_VALIDATION_SPLIT,
    TRAIN_DATASET_PATH,
    TEST_DATASET_PATH
)
from preprocessing.load_dataset import load_unsw_nb15_datasets
from preprocessing.pipeline import PreprocessingPipeline
from training.class_weights import compute_multiclass_weights
from models.hybrid_model import build_hybrid_ids_model
from training.evaluate_model import evaluate_predictions, plot_confusion_matrix, save_evaluation_artifacts

logger = get_logger("TrainModel")

def plot_training_history(history_dict: Dict[str, list], save_path: Optional[Path] = None) -> Path:
    """Generates dual-axis training and validation loss/accuracy curves."""
    save_path = save_path or (PLOTS_DIR / "training_curves.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    epochs = range(1, len(history_dict.get("loss", [])) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    
    # Loss Plot
    ax1.plot(epochs, history_dict.get("loss", []), "o-", color="#3B82F6", label="Training Loss", linewidth=2)
    ax1.plot(epochs, history_dict.get("val_loss", []), "s--", color="#EF4444", label="Validation Loss", linewidth=2)
    ax1.set_title("Cross-Entropy Loss Progression", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    # Accuracy Plot
    ax2.plot(epochs, history_dict.get("accuracy", []), "o-", color="#10B981", label="Training Accuracy", linewidth=2)
    ax2.plot(epochs, history_dict.get("val_accuracy", []), "s--", color="#8B5CF6", label="Validation Accuracy", linewidth=2)
    ax2.set_title("Classification Accuracy Progression", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Accuracy", fontsize=10)
    ax2.legend(loc="lower right")
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved training history curves to: {save_path}")
    return save_path

def train_hybrid_ids(
    train_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    filters: int = DEFAULT_CNN_FILTERS,
    lstm_units: int = DEFAULT_LSTM_UNITS,
    dense_units: int = DEFAULT_DENSE_UNITS,
    dropout_rate: float = DEFAULT_DROPOUT_RATE,
    validation_split: float = DEFAULT_VALIDATION_SPLIT,
    use_class_weights: bool = True,
    sample_subsample: Optional[int] = None,
    save_artifacts: bool = True
) -> Tuple[keras.Model, PreprocessingPipeline, Dict[str, Any]]:
    """
    Executes the complete training workflow.
    
    Parameters:
        sample_subsample: If provided, trains on a representative stratified subsample (for fast demo or verification).
    """
    start_total_time = time.time()
    logger.info("Initializing Hybrid IDS training pipeline...")

    # 1. Load Data if not provided
    if train_df is None or test_df is None:
        train_df, test_df = load_unsw_nb15_datasets()

    # If subsample requested (e.g. for development / verification mode)
    if sample_subsample is not None and len(train_df) > sample_subsample:
        from sklearn.model_selection import train_test_split
        logger.info(f"Stratified subsampling training dataset to {sample_subsample:,} records.")
        train_df, _ = train_test_split(
            train_df, train_size=sample_subsample, stratify=train_df["attack_cat"], random_state=42
        )
        test_sub = min(len(test_df), sample_subsample // 2)
        test_df, _ = train_test_split(
            test_df, train_size=test_sub, stratify=test_df["attack_cat"], random_state=42
        )
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        logger.info(f"Subsampled train size: {len(train_df):,}, test size: {len(test_df):,}")

    # 2. Fit and Transform Preprocessing Pipeline
    pipeline = PreprocessingPipeline(sequence_length=sequence_length)
    X_train_seq, y_train_seq, X_test_seq, y_test_seq = pipeline.fit_transform_splits(
        train_df, test_df, stride=1
    )

    num_features = X_train_seq.shape[2]
    input_shape = (sequence_length, num_features)

    # 3. Calculate Class Weights
    class_weights = None
    if use_class_weights:
        class_weights = compute_multiclass_weights(y_train_seq, num_classes=10)

    # 4. Build Model
    model = build_hybrid_ids_model(
        input_shape=input_shape,
        num_classes=10,
        filters=filters,
        lstm_units=lstm_units,
        dense_units=dense_units,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate
    )

    # 5. Training Callbacks
    model_save_path = MODELS_DIR / "hybrid_ids_model.keras"
    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1
        )
    ]

    # 6. Fit Model
    logger.info(f"Starting model training on {len(X_train_seq):,} sequences (Batch={batch_size}, Max Epochs={epochs})...")
    train_start_time = time.time()
    history = model.fit(
        X_train_seq,
        y_train_seq,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        class_weight=class_weights if use_class_weights else None,
        callbacks=cb_list,
        verbose=1
    )
    training_duration_s = time.time() - train_start_time
    logger.info(f"Training completed in {training_duration_s:.2f} seconds.")

    # 7. Evaluate on Official Test Sequences
    logger.info("Evaluating trained model on testing sequences...")
    eval_start_time = time.time()
    y_pred_probs = model.predict(X_test_seq, batch_size=batch_size, verbose=0)
    eval_latency_ms = (time.time() - eval_start_time) * 1000.0

    eval_results = evaluate_predictions(y_test_seq, y_pred_probs, latency_ms=eval_latency_ms)
    eval_results["training_time_seconds"] = round(training_duration_s, 2)
    eval_results["epochs_trained"] = len(history.history["loss"])
    eval_results["hyperparameters"] = {
        "sequence_length": sequence_length,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "filters": filters,
        "lstm_units": lstm_units,
        "dense_units": dense_units,
        "dropout_rate": dropout_rate,
        "class_weighted": use_class_weights
    }

    # 8. Save Artifacts
    if save_artifacts:
        pipeline.save()
        model.save(model_save_path)
        logger.info(f"Saved trained model to: {model_save_path}")

        # Save History
        history_clean = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        save_json(history_clean, REPORTS_DIR / "training_history.json")
        plot_training_history(history_clean)
        
        # Save Confusion Matrix & Evaluation
        cm = np.array(eval_results["confusion_matrix"])
        plot_confusion_matrix(cm)
        save_evaluation_artifacts(eval_results)

    total_time_s = time.time() - start_total_time
    logger.info(f"Full training and evaluation pipeline completed in {total_time_s:.2f} seconds.")
    return model, pipeline, eval_results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Explainable Hybrid IDS Model.")
    parser.add_argument("--quick", action="store_true", help="Run quick verification with subsample.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    subsample = 15000 if args.quick else None
    train_hybrid_ids(sample_subsample=subsample, epochs=args.epochs, batch_size=args.batch_size)
