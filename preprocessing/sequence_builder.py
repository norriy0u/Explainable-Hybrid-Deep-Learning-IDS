import sys
from pathlib import Path
_base_dir = Path(__file__).resolve().parent
while _base_dir.name != 'Explainable_Hybrid_IDS' and _base_dir.parent != _base_dir:
    _base_dir = _base_dir.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

"""
Sequence Construction Module for Temporal / Flow-Ordered Deep Learning IDS.
Constructs sliding-window sequences of length T with final-observation target labels.
Preserves dataset boundaries and guarantees no data leakage between splits.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
from utils.logger import get_logger

logger = get_logger("SequenceBuilder")

def create_sequences(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    sequence_length: int = 5,
    stride: int = 1
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Constructs sliding window sequences of shape (M, sequence_length, D).
    
    Parameters:
        X: Feature matrix of shape (N, D).
        y: Target label array of shape (N,) or (N, K). Can be None for inference.
        sequence_length: Number of consecutive observations per sequence window (T).
        stride: Step size between consecutive sequence windows.
        
    Returns:
        X_seq: 3D array of shape (M, sequence_length, D).
        y_seq: 1D or 2D array of targets corresponding to the final observation in each window.
    """
    N, D = X.shape
    if N < sequence_length:
        raise ValueError(
            f"Input records count ({N}) is smaller than requested sequence_length ({sequence_length}). "
            f"Cannot construct full sequence windows without padding."
        )

    num_sequences = (N - sequence_length) // stride + 1
    
    # Vectorized sliding window sequence creation
    X_seq = np.zeros((num_sequences, sequence_length, D), dtype=X.dtype)
    y_seq = None
    if y is not None:
        if y.ndim == 1:
            y_seq = np.zeros((num_sequences,), dtype=y.dtype)
        else:
            y_seq = np.zeros((num_sequences, y.shape[1]), dtype=y.dtype)

    for i in range(num_sequences):
        start_idx = i * stride
        end_idx = start_idx + sequence_length
        X_seq[i] = X[start_idx:end_idx]
        if y is not None:
            # Final observation's label represents the ground-truth state at time T
            y_seq[i] = y[end_idx - 1]

    logger.info(
        f"Generated {num_sequences:,} sequences of shape (T={sequence_length}, D={D}) "
        f"with stride={stride} from {N:,} raw records."
    )
    return X_seq, y_seq

def pad_or_replicate_for_single_inference(
    X: np.ndarray,
    sequence_length: int = 5
) -> np.ndarray:
    """
    Pads or replicates a short stream (e.g. 1 to T-1 records) to form a valid (1, T, D) sequence.
    If 1 sample is provided, replicates across the T time steps (representing a sustained connection flow).
    """
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    n_samples, D = X.shape
    if n_samples >= sequence_length:
        # Take the most recent sequence_length observations
        return X[-sequence_length:].reshape(1, sequence_length, D)
    
    # Replicate the available samples to reach sequence_length
    repeats = int(np.ceil(sequence_length / n_samples))
    replicated = np.tile(X, (repeats, 1))[:sequence_length]
    return replicated.reshape(1, sequence_length, D)

def validate_sequence_integrity(
    X_seq: np.ndarray,
    y_seq: Optional[np.ndarray],
    expected_t: int,
    expected_d: int
) -> Dict[str, Any]:
    """Validates sequence dimensional integrity and finiteness."""
    is_valid = True
    errors = []

    if X_seq.ndim != 3:
        is_valid = False
        errors.append(f"Expected 3D sequence array (B, T, D), got {X_seq.ndim}D.")
    else:
        if X_seq.shape[1] != expected_t:
            is_valid = False
            errors.append(f"Expected sequence length T={expected_t}, got T={X_seq.shape[1]}.")
        if X_seq.shape[2] != expected_d:
            is_valid = False
            errors.append(f"Expected feature dimension D={expected_d}, got D={X_seq.shape[2]}.")

    if not np.isfinite(X_seq).all():
        is_valid = False
        errors.append("Sequence contains infinite or NaN values.")

    if y_seq is not None:
        if len(X_seq) != len(y_seq):
            is_valid = False
            errors.append(f"Mismatch: {len(X_seq)} sequences vs {len(y_seq)} labels.")

    return {
        "is_valid": is_valid,
        "num_sequences": len(X_seq),
        "shape": list(X_seq.shape),
        "errors": errors
    }
