import sys
from pathlib import Path
_base_dir = Path(__file__).resolve().parent
while _base_dir.name != 'Explainable_Hybrid_IDS' and _base_dir.parent != _base_dir:
    _base_dir = _base_dir.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

"""
Unit Tests for Sequence Construction and Sliding Windows.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from preprocessing.sequence_builder import create_sequences, pad_or_replicate_for_single_inference, validate_sequence_integrity

def test_sequence_construction():
    # 10 records, 4 features
    N, D = 10, 4
    X = np.arange(N * D, dtype=np.float32).reshape(N, D)
    y = np.arange(N, dtype=np.int32)
    
    T = 5
    X_seq, y_seq = create_sequences(X, y, sequence_length=T, stride=1)
    
    # Expected number of windows: 10 - 5 + 1 = 6
    assert X_seq.shape == (6, 5, 4)
    assert y_seq.shape == (6,)
    
    # Final observation label check: Window 0 spans rows 0..4 -> label is y[4]
    assert y_seq[0] == y[4]
    assert y_seq[5] == y[9]
    
    val = validate_sequence_integrity(X_seq, y_seq, expected_t=5, expected_d=4)
    assert val["is_valid"] is True
    print("test_sequence_construction passed!")

def test_single_sample_replication():
    single_obs = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    seq = pad_or_replicate_for_single_inference(single_obs, sequence_length=5)
    assert seq.shape == (1, 5, 4)
    assert np.all(seq[0, 0] == seq[0, 4])
    print("test_single_sample_replication passed!")

if __name__ == "__main__":
    test_sequence_construction()
    test_single_sample_replication()
    print("ALL SEQUENCE TESTS PASSED!")
