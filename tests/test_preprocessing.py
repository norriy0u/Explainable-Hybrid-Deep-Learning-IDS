import sys
from pathlib import Path
_base_dir = Path(__file__).resolve().parent
while _base_dir.name != 'Explainable_Hybrid_IDS' and _base_dir.parent != _base_dir:
    _base_dir = _base_dir.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

"""
Unit Tests for Preprocessing and Data Cleaning Modules.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from preprocessing.clean_data import standardize_attack_categories, clean_network_dataframe
from preprocessing.encode_features import CategoricalFeatureEncoder, TargetLabelEncoder
from preprocessing.feature_scaling import NumericalFeatureScaler

def test_standardize_attack_categories():
    s = pd.Series([" dos ", "backdoors", "NORMAL", "Exploits", "UNKNOWN_ATTACK", "worms"])
    cleaned = standardize_attack_categories(s)
    assert cleaned.iloc[0] == "DoS"
    assert cleaned.iloc[1] == "Backdoor"
    assert cleaned.iloc[2] == "Normal"
    assert cleaned.iloc[3] == "Exploits"
    assert cleaned.iloc[5] == "Worms"
    print("test_standardize_attack_categories passed!")

def test_categorical_and_target_encoding():
    df_train = pd.DataFrame({
        "proto": ["tcp", "udp", "tcp"],
        "service": ["http", "dns", "-"],
        "state": ["FIN", "CON", "FIN"],
        "attack_cat": ["Normal", "DoS", "Exploits"]
    })
    encoder = CategoricalFeatureEncoder(["proto", "service", "state"])
    encoded_train = encoder.fit_transform(df_train)
    assert encoded_train["proto"].dtype in [np.int32, np.int64, int]
    
    # Test unknown category handling in test
    df_test = pd.DataFrame({
        "proto": ["icmp_new"], # Unseen protocol
        "service": ["http"],
        "state": ["FIN"]
    })
    encoded_test = encoder.transform(df_test)
    assert encoded_test["proto"].iloc[0] == encoder.unknown_indices["proto"]
    
    # Test target encoding (10 classes)
    t_encoder = TargetLabelEncoder()
    targets = pd.Series(["Normal", "DoS", "Worms", "Exploits"])
    indices = t_encoder.transform(targets)
    assert indices[0] == 0  # Normal
    assert indices[1] == 3  # DoS
    assert indices[2] == 9  # Worms
    assert indices[3] == 4  # Exploits
    
    one_hot = t_encoder.to_categorical(indices)
    assert one_hot.shape == (4, 10)
    assert one_hot[1, 3] == 1.0 # DoS is class 3
    print("test_categorical_and_target_encoding passed!")

def test_numerical_scaling():
    df = pd.DataFrame({
        "sbytes": [100.0, 200.0, 300.0],
        "dbytes": [10.0, 20.0, 30.0]
    })
    scaler = NumericalFeatureScaler()
    scaled = scaler.fit_transform(df)
    assert scaled.shape == (3, 2)
    assert np.allclose(scaled.mean(axis=0), 0.0, atol=1e-5)
    print("test_numerical_scaling passed!")

if __name__ == "__main__":
    test_standardize_attack_categories()
    test_categorical_and_target_encoding()
    test_numerical_scaling()
    print("ALL PREPROCESSING TESTS PASSED!")
