import sys
from pathlib import Path
_base_dir = Path(__file__).resolve().parent
while _base_dir.name != 'Explainable_Hybrid_IDS' and _base_dir.parent != _base_dir:
    _base_dir = _base_dir.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

"""
Class Imbalance Handling Module for Multiclass Network Intrusion Detection.
Calculates balanced class weights according to the research formula:
w_c = N / (K * N_c)
Ensures minority attack classes (Worms, Backdoor, Shellcode, Analysis) receive equitable loss weighting.
"""

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from utils.logger import get_logger
from config.config import LABEL_MAPPING, CLASS_NAMES

logger = get_logger("ClassWeights")

def compute_multiclass_weights(
    y_train: Union[np.ndarray, pd.Series],
    num_classes: int = 10
) -> Dict[int, float]:
    """
    Computes balanced class weights:
    w_c = N / (K * N_c)
    
    Parameters:
        y_train: Integer class indices (N,) or one-hot array (N, K).
        num_classes: Total number of classes (default 10).
        
    Returns:
        Dict mapping integer class index to its computed weight float.
    """
    if isinstance(y_train, pd.Series):
        y_indices = y_train.values
    elif isinstance(y_train, np.ndarray):
        if y_train.ndim == 2:
            y_indices = np.argmax(y_train, axis=1)
        else:
            y_indices = y_train
    else:
        y_indices = np.array(y_train)

    N = len(y_indices)
    K = num_classes
    
    unique_classes, counts = np.unique(y_indices, return_counts=True)
    counts_dict = dict(zip(unique_classes, counts))
    
    class_weights = {}
    distribution_report = {}
    
    logger.info(f"Computing class weights for {N:,} training samples across {K} classes:")
    for c in range(K):
        N_c = counts_dict.get(c, 0)
        class_name = LABEL_MAPPING.get(c, f"Class_{c}")
        if N_c > 0:
            # w_c = N / (K * N_c)
            weight = float(N / (K * N_c))
        else:
            # If a class has zero training samples, assign default weight of 1.0
            weight = 1.0
            
        class_weights[c] = round(weight, 4)
        distribution_report[class_name] = {
            "class_id": c,
            "sample_count": int(N_c),
            "percentage": round((N_c / N) * 100, 3) if N > 0 else 0.0,
            "weight": round(weight, 4)
        }
        logger.info(
            f"Class {c:02d} ({class_name:<14}): Count={N_c:6,d} "
            f"({distribution_report[class_name]['percentage']:6.2f}%) -> Weight={weight:.4f}"
        )
        
    return class_weights

def get_class_weight_summary(class_weights: Dict[int, float]) -> pd.DataFrame:
    """Returns a formatted summary dataframe of class weights for display in the dashboard."""
    records = []
    for c, w in class_weights.items():
        records.append({
            "Class ID": c,
            "Attack Category": LABEL_MAPPING.get(c, f"Class {c}"),
            "Assigned Weight": w
        })
    return pd.DataFrame(records)
