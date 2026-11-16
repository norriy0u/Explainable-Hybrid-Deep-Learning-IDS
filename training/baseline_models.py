import sys
from pathlib import Path
_base_dir = Path(__file__).resolve().parent
while 'config' not in [p.name for p in _base_dir.iterdir() if p.is_dir()] and _base_dir.parent != _base_dir:
    _base_dir = _base_dir.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

"""
Ablation Study and Baseline Model Comparison Module.
Compares:
  1. CNN Baseline
  2. BiLSTM Baseline
  3. CNN-BiLSTM (without Attention)
  4. CNN-BiLSTM-Attention (unweighted)
  5. Full Proposed Hybrid (with Class Weighting)
Also conducts sequence length ablation across T in [3, 5, 10].
"""

import os
os.environ["KERAS_BACKEND"] = "torch"
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import keras

from utils.logger import get_logger
from utils.helpers import save_json, load_json
from config.config import (
    EXPERIMENT_RESULTS_DIR,
    PLOTS_DIR,
    CLASS_NAMES,
    LABEL_MAPPING,
    DEMO_ATTACK_CLASS
)
from preprocessing.load_dataset import load_unsw_nb15_datasets
from preprocessing.pipeline import PreprocessingPipeline
from training.class_weights import compute_multiclass_weights
from models.cnn_model import build_cnn_model
from models.bilstm_model import build_bilstm_model
from models.cnn_bilstm import build_cnn_bilstm_model
from models.hybrid_model import build_hybrid_ids_model
from training.evaluate_model import evaluate_predictions

logger = get_logger("AblationStudy")

def plot_ablation_comparison(results_data: Dict[str, Any], save_path: Optional[Path] = None) -> Path:
    """Generates comparison bar charts across the five model architectures."""
    save_path = save_path or (PLOTS_DIR / "model_comparison.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    models = list(results_data.keys())
    accuracies = [results_data[m]["accuracy"] * 100 for m in models]
    macro_f1s = [results_data[m]["macro_f1"] * 100 for m in models]
    dos_f1s = [results_data[m]["dos_performance"].get("f1_score", 0) * 100 for m in models]
    latencies = [results_data[m].get("inference_latency_ms_per_sample", 0) * 1000 for m in models] # microsec

    x = np.arange(len(models))
    width = 0.25

    fig, ax1 = plt.subplots(figsize=(13, 6), dpi=150)
    sns.set_theme(style="whitegrid")
    
    rects1 = ax1.bar(x - width, accuracies, width, label="Accuracy (%)", color="#3B82F6")
    rects2 = ax1.bar(x, macro_f1s, width, label="Macro F1 (%)", color="#10B981")
    rects3 = ax1.bar(x + width, dos_f1s, width, label="DoS Attack F1 (%)", color="#EF4444")

    ax1.set_ylabel("Performance (%)", fontsize=12, fontweight="bold")
    ax1.set_title("Architectural Ablation Study Comparison (UNSW-NB15)", fontsize=14, fontweight="bold", pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=15, ha="right", fontsize=10, fontweight="medium")
    ax1.legend(loc="lower right", framealpha=0.9)
    ax1.set_ylim(0, 105)

    # Add values on top of bars
    for rect in rects1 + rects2 + rects3:
        height = rect.get_height()
        ax1.annotate(f"{height:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved ablation comparison chart to: {save_path}")
    return save_path

def plot_sequence_length_comparison(results_data: Dict[str, Any], save_path: Optional[Path] = None) -> Path:
    """Generates comparison chart across sequence lengths T in [3, 5, 10]."""
    save_path = save_path or (PLOTS_DIR / "sequence_length_comparison.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    lengths = [str(k) for k in results_data.keys()]
    accs = [results_data[k]["accuracy"] * 100 for k in results_data.keys()]
    macro_f1s = [results_data[k]["macro_f1"] * 100 for k in results_data.keys()]
    dos_f1s = [results_data[k]["dos_performance"].get("f1_score", 0) * 100 for k in results_data.keys()]

    plt.figure(figsize=(9, 5), dpi=150)
    plt.plot(lengths, accs, "o-", color="#3B82F6", label="Accuracy (%)", linewidth=2.5, markersize=8)
    plt.plot(lengths, macro_f1s, "s--", color="#10B981", label="Macro F1-Score (%)", linewidth=2.5, markersize=8)
    plt.plot(lengths, dos_f1s, "^-.", color="#EF4444", label="DoS Attack F1-Score (%)", linewidth=2.5, markersize=8)

    plt.title("Sequence Length (T) Impact on Detection Performance", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Sequence Length Window (T)", fontsize=11, fontweight="medium")
    plt.ylabel("Score (%)", fontsize=11, fontweight="medium")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved sequence length comparison plot to: {save_path}")
    return save_path

def run_ablation_study(
    subsample_size: int = 15000,
    epochs: int = 4,
    batch_size: int = 128
) -> Dict[str, Any]:
    """
    Executes a structured ablation experiment comparing the 5 model configurations
    and sequence length windows under strict experimental control.
    """
    logger.info("Starting ablation study across model architectures...")
    train_df, test_df = load_unsw_nb15_datasets()

    # Stratified subsampling for controlled ablation execution
    if subsample_size and len(train_df) > subsample_size:
        from sklearn.model_selection import train_test_split
        train_df, _ = train_test_split(
            train_df, train_size=subsample_size, stratify=train_df["attack_cat"], random_state=42
        )
        test_sub = min(len(test_df), subsample_size // 2)
        test_df, _ = train_test_split(
            test_df, train_size=test_sub, stratify=test_df["attack_cat"], random_state=42
        )
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

    # 1. Pipeline with T=5
    pipeline = PreprocessingPipeline(sequence_length=5)
    X_train, y_train, X_test, y_test = pipeline.fit_transform_splits(train_df, test_df)
    class_weights = compute_multiclass_weights(y_train, num_classes=10)
    
    input_shape = (5, X_train.shape[2])
    ablation_results = {}

    architectures = [
        ("1. CNN Baseline", lambda: build_cnn_model(input_shape=input_shape), False),
        ("2. BiLSTM Baseline", lambda: build_bilstm_model(input_shape=input_shape), False),
        ("3. CNN + BiLSTM", lambda: build_cnn_bilstm_model(input_shape=input_shape), False),
        ("4. CNN + BiLSTM + Attention", lambda: build_hybrid_ids_model(input_shape=input_shape), False),
        ("5. Proposed Hybrid (Class Weighted)", lambda: build_hybrid_ids_model(input_shape=input_shape), True),
    ]

    for model_name, builder_fn, use_weights in architectures:
        logger.info(f"Training ablation model: {model_name}...")
        model = builder_fn()
        t0 = time.time()
        model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.15,
            class_weight=class_weights if use_weights else None,
            verbose=0
        )
        train_time = time.time() - t0
        
        # Inference & Evaluation
        t1 = time.time()
        preds = model.predict(X_test, batch_size=batch_size, verbose=0)
        eval_time = (time.time() - t1) * 1000.0
        
        metrics = evaluate_predictions(y_test, preds, latency_ms=eval_time)
        metrics["training_time_s"] = round(train_time, 2)
        ablation_results[model_name] = metrics
        logger.info(
            f"Finished {model_name} -> Acc: {metrics['accuracy']:.4f}, "
            f"Macro F1: {metrics['macro_f1']:.4f}, DoS F1: {metrics['dos_performance'].get('f1_score', 0):.4f}"
        )

    # Save ablation results
    save_json(ablation_results, EXPERIMENT_RESULTS_DIR / "ablation_study.json")
    plot_ablation_comparison(ablation_results)

    # 2. Sequence Length Ablation (T in [3, 5, 10])
    logger.info("Conducting Sequence Length Ablation (T=3, 5, 10)...")
    seq_results = {}
    for T in [3, 5, 10]:
        logger.info(f"Testing Sequence Length T={T}...")
        p_t = PreprocessingPipeline(sequence_length=T)
        X_tr, y_tr, X_te, y_te = p_t.fit_transform_splits(train_df, test_df)
        weights_t = compute_multiclass_weights(y_tr, num_classes=10)
        
        model_t = build_hybrid_ids_model(input_shape=(T, X_tr.shape[2]))
        model_t.fit(
            X_tr, y_tr,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.15,
            class_weight=weights_t,
            verbose=0
        )
        preds_t = model_t.predict(X_te, batch_size=batch_size, verbose=0)
        seq_results[f"T = {T}"] = evaluate_predictions(y_te, preds_t)
        
    save_json(seq_results, EXPERIMENT_RESULTS_DIR / "sequence_length_study.json")
    plot_sequence_length_comparison(seq_results)

    logger.info("Ablation study and Sequence Length experiments complete!")
    return {
        "ablation_study": ablation_results,
        "sequence_length_study": seq_results
    }

if __name__ == "__main__":
    run_ablation_study(subsample_size=12000, epochs=3)
