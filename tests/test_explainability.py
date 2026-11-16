import sys
from pathlib import Path
_base_dir = Path(__file__).resolve().parent
while 'config' not in [p.name for p in _base_dir.iterdir() if p.is_dir()] and _base_dir.parent != _base_dir:
    _base_dir = _base_dir.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

"""
Unit Tests for Explainable AI (SHAP) Module.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
from models.hybrid_model import build_hybrid_ids_model
from explainability.shap_explainer import IDSShapExplainer

def test_shap_explanation_generation():
    T = 5
    D = 10
    feature_names = [f"feat_{i}" for i in range(D)]
    feature_names[0] = "sbytes"
    feature_names[1] = "rate"
    
    # Simple model for fast test
    model = build_hybrid_ids_model(input_shape=(T, D), num_classes=10, filters=16, lstm_units=16, dense_units=16)
    
    bg = np.random.randn(15, T, D).astype(np.float32)
    explainer = IDSShapExplainer(model, bg, feature_names, max_background=10)
    
    sample = np.random.randn(1, T, D).astype(np.float32)
    res = explainer.explain_sequence_prediction(sample, target_class=3, top_k=5)
    
    assert "predicted_class_name" in res
    assert "predicted_confidence" in res
    assert res["explained_class_id"] == 3
    assert len(res["top_features"]) == 5
    assert "summary_text" in res
    
    top1 = res["top_features"][0]
    assert "feature" in top1
    assert "shap_value" in top1
    assert "cyber_context" in top1
    print("SHAP explainer test passed successfully!")

if __name__ == "__main__":
    test_shap_explanation_generation()
    print("ALL EXPLAINABILITY TESTS PASSED!")
