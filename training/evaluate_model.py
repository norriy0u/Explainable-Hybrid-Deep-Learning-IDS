import sys
from pathlib import Path
_base_dir = Path(__file__).resolve().parent
while 'config' not in [p.name for p in _base_dir.iterdir() if p.is_dir()] and _base_dir.parent != _base_dir:
    _base_dir = _base_dir.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

"""
Model Evaluation and Diagnostic Metrics Module.
Calculates Accuracy, Precision, Recall, Macro/Weighted F1, Per-Class metrics,
Confusion Matrix heatmap, and inference latency benchmarks.
"""

import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from utils.logger import get_logger
from utils.helpers import save_json
from config.config import (
    CLASS_NAMES,
    LABEL_MAPPING,
    PLOTS_DIR,
    REPORTS_DIR,
    DEMO_ATTACK_ID,
    DEMO_ATTACK_CLASS
)

logger = get_logger("ModelEvaluation")

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    latency_ms: Optional[float] = None
) -> Dict[str, Any]:
    """
    Evaluates model predictions against true labels.
    
    Parameters:
        y_true: 1D array of true class integers or 2D one-hot array (N, 10).
        y_pred_probs: 2D array of predicted class probabilities (N, 10).
        latency_ms: Measured total inference latency in milliseconds.
    """
    if y_true.ndim == 2:
        y_true_indices = np.argmax(y_true, axis=1)
    else:
        y_true_indices = y_true.astype(int)
        
    y_pred_indices = np.argmax(y_pred_probs, axis=1)
    N = len(y_true_indices)

    # Core Metrics
    accuracy = float(accuracy_score(y_true_indices, y_pred_indices))
    macro_precision = float(precision_score(y_true_indices, y_pred_indices, average="macro", zero_division=0))
    weighted_precision = float(precision_score(y_true_indices, y_pred_indices, average="weighted", zero_division=0))
    macro_recall = float(recall_score(y_true_indices, y_pred_indices, average="macro", zero_division=0))
    weighted_recall = float(recall_score(y_true_indices, y_pred_indices, average="weighted", zero_division=0))
    macro_f1 = float(f1_score(y_true_indices, y_pred_indices, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true_indices, y_pred_indices, average="weighted", zero_division=0))

    # Per-Class Metrics
    per_class_p = precision_score(y_true_indices, y_pred_indices, average=None, labels=range(10), zero_division=0)
    per_class_r = recall_score(y_true_indices, y_pred_indices, average=None, labels=range(10), zero_division=0)
    per_class_f = f1_score(y_true_indices, y_pred_indices, average=None, labels=range(10), zero_division=0)

    per_class_metrics = {}
    for i in range(10):
        c_name = LABEL_MAPPING[i]
        support = int(np.sum(y_true_indices == i))
        per_class_metrics[c_name] = {
            "class_id": i,
            "precision": float(round(per_class_p[i], 4)),
            "recall": float(round(per_class_r[i], 4)),
            "f1_score": float(round(per_class_f[i], 4)),
            "support": support
        }

    # Confusion Matrix
    cm = confusion_matrix(y_true_indices, y_pred_indices, labels=range(10))

    # DoS Specific Focus
    dos_metrics = per_class_metrics.get(DEMO_ATTACK_CLASS, {})

    results = {
        "num_test_samples": N,
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "weighted_precision": round(weighted_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "weighted_recall": round(weighted_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "inference_latency_ms_total": round(latency_ms, 2) if latency_ms is not None else None,
        "inference_latency_ms_per_sample": round(latency_ms / N, 4) if (latency_ms is not None and N > 0) else None,
        "dos_performance": dos_metrics,
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": cm.tolist()
    }

    logger.info(
        f"Evaluation Complete - Accuracy: {results['accuracy']:.4f}, "
        f"Macro F1: {results['macro_f1']:.4f}, Weighted F1: {results['weighted_f1']:.4f}, "
        f"DoS F1: {dos_metrics.get('f1_score', 0):.4f}"
    )
    return results

def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: Optional[Path] = None,
    normalize: bool = False
) -> Path:
    """Renders and saves an academic-grade confusion matrix heatmap."""
    save_path = save_path or (PLOTS_DIR / "confusion_matrix.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    cm_display = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] if normalize else cm
    cm_display = np.nan_to_num(cm_display)

    plt.figure(figsize=(10, 8), dpi=150)
    sns.set_theme(style="darkgrid")
    
    fmt = ".2f" if normalize else "d"
    ax = sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar_kws={"label": "Normalized Proportion" if normalize else "Observation Count"}
    )
    
    plt.title("Multiclass Confusion Matrix - Hybrid IDS (UNSW-NB15)", fontsize=14, pad=15, fontweight="bold")
    plt.xlabel("Predicted Attack Category", fontsize=12, labelpad=10)
    plt.ylabel("Ground-Truth Attack Category", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved confusion matrix plot to: {save_path}")
    return save_path

def save_evaluation_artifacts(results: Dict[str, Any]) -> None:
    """Saves evaluation results dictionary and formatted classification report to disk."""
    json_path = REPORTS_DIR / "evaluation_results.json"
    save_json(results, json_path)
    
    # Text report
    txt_path = REPORTS_DIR / "classification_report.txt"
    lines = [
        "=" * 65,
        "EXPLAINABLE HYBRID IDS - TEST EVALUATION REPORT",
        "=" * 65,
        f"Total Test Sequences:    {results['num_test_samples']:,}",
        f"Overall Accuracy:        {results['accuracy'] * 100:.2f}%",
        f"Macro F1-Score:          {results['macro_f1'] * 100:.2f}%",
        f"Weighted F1-Score:       {results['weighted_f1'] * 100:.2f}%",
        f"Macro Precision:         {results['macro_precision'] * 100:.2f}%",
        f"Macro Recall:            {results['macro_recall'] * 100:.2f}%",
        "-" * 65,
        f"{'Category':<16} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<8}",
        "-" * 65
    ]
    for c_name, m in results["per_class_metrics"].items():
        lines.append(
            f"{c_name:<16} {m['precision']*100:6.2f}%      {m['recall']*100:6.2f}%      {m['f1_score']*100:6.2f}%      {m['support']:<8}"
        )
    lines.append("=" * 65)
    
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        
    logger.info(f"Saved evaluation reports to: {json_path} and {txt_path}")
