import sys
from pathlib import Path
_base_dir = Path(__file__).resolve().parent
while 'config' not in [p.name for p in _base_dir.iterdir() if p.is_dir()] and _base_dir.parent != _base_dir:
    _base_dir = _base_dir.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

"""
Unit Tests for Deep Learning Models: CNN, BiLSTM, CNN-BiLSTM, Attention, and Proposed Hybrid.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras import ops

from models.cnn_model import build_cnn_model
from models.bilstm_model import build_bilstm_model
from models.cnn_bilstm import build_cnn_bilstm_model
from models.hybrid_model import build_hybrid_ids_model

def to_np(t):
    return ops.convert_to_numpy(t)

def test_model_architectures():
    batch_size = 4
    T = 5
    D = 42
    dummy_input = np.random.randn(batch_size, T, D).astype(np.float32)
    
    # 1. Test CNN
    cnn = build_cnn_model(input_shape=(T, D), num_classes=10)
    out_cnn = to_np(cnn(dummy_input, training=False))
    assert out_cnn.shape == (batch_size, 10)
    assert np.allclose(out_cnn.sum(axis=1), 1.0, atol=1e-5)
    print("CNN Model passed!")

    # 2. Test BiLSTM
    bilstm = build_bilstm_model(input_shape=(T, D), num_classes=10)
    out_bilstm = to_np(bilstm(dummy_input, training=False))
    assert out_bilstm.shape == (batch_size, 10)
    assert np.allclose(out_bilstm.sum(axis=1), 1.0, atol=1e-5)
    print("BiLSTM Model passed!")

    # 3. Test CNN-BiLSTM
    cnn_bilstm = build_cnn_bilstm_model(input_shape=(T, D), num_classes=10)
    out_cb = to_np(cnn_bilstm(dummy_input, training=False))
    assert out_cb.shape == (batch_size, 10)
    assert np.allclose(out_cb.sum(axis=1), 1.0, atol=1e-5)
    print("CNN-BiLSTM Model passed!")

    # 4. Test Proposed Hybrid Architecture
    hybrid = build_hybrid_ids_model(input_shape=(T, D), num_classes=10)
    out_hybrid = to_np(hybrid(dummy_input, training=False))
    assert out_hybrid.shape == (batch_size, 10)
    assert np.allclose(out_hybrid.sum(axis=1), 1.0, atol=1e-5)
    print("Proposed Hybrid Model passed!")

    # 5. Test Proposed Hybrid with Attention Weights Output
    hybrid_attn = build_hybrid_ids_model(input_shape=(T, D), num_classes=10, return_attention=True)
    preds, weights = hybrid_attn(dummy_input, training=False)
    preds_np = to_np(preds)
    weights_np = to_np(weights)
    assert preds_np.shape == (batch_size, 10)
    assert weights_np.shape == (batch_size, T)
    # Attention weights must sum to 1.0 across time steps
    assert np.allclose(weights_np.sum(axis=1), 1.0, atol=1e-5)
    print("Proposed Hybrid with Attention Weights passed!")

if __name__ == "__main__":
    test_model_architectures()
    print("ALL MODEL ARCHITECTURE TESTS PASSED SUCCESSFULLY!")
