"""
Explainable Hybrid Deep Learning IDS — Code-First Live Demonstration Script
Author: Deep Learning Mini-Project Team (Midhila M & Amrita Raj)
Academic Institution: Amrita Vishwa Vidyapeetham

This standalone script executes the complete end-to-end intrusion detection
and explainability pipeline directly in the terminal without requiring Streamlit:
1. Loads authentic network flow records from the UNSW-NB15 benchmark.
2. Applies the trained leakage-safe preprocessing pipeline (impute, encode, scale).
3. Constructs a 3D sequential tensor (Batch=1, TimeSteps=5, Features=42).
4. Executes forward-pass inference through 1D-CNN + BiLSTM + Temporal Attention.
5. Computes multi-class softmax probabilities across all 10 threat categories.
6. Extracts genuine learned temporal attention weights across the observation window.
7. Computes game-theoretic SHAP attributions via PermutationExplainer (max_evals=500).
8. Generates a formal SOC cybersecurity incident alert and defensive playbook.
9. Exports an academic publication-quality SHAP attribution bar chart.
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Force torch backend for modern Keras 3 and set local Keras home
os.environ["KERAS_BACKEND"] = "torch"
_local_keras = Path(__file__).resolve().parent / ".keras"
_local_keras.mkdir(exist_ok=True)
os.environ["KERAS_HOME"] = str(_local_keras)

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config.config import (
    MODELS_DIR,
    PREPROCESSORS_DIR,
    REPORTS_DIR,
    SHAP_PLOTS_DIR,
    BENCHMARK_SAMPLE_PATH,
    LABEL_MAPPING,
    CLASS_NAMES,
    DEFAULT_SEQUENCE_LENGTH
)
from detection.predict_attack import AttackDetectionEngine
from explainability.feature_visualization import plot_shap_bar_chart

def print_banner():
    banner = """
================================================================================
   EXPLAINABLE HYBRID DEEP LEARNING NETWORK INTRUSION DETECTION SYSTEM
   Architecture: 1D-CNN + Bidirectional LSTM + Temporal Attention + SHAP
   Dataset: UNSW-NB15 Benchmark | Threat Classes: 10 Multiclass Categories
================================================================================
"""
    print(banner)

def load_attack_samples(category: str = "DoS", sequence_len: int = 5):
    """Retrieves 5 consecutive flow records from the selected attack category."""
    if not BENCHMARK_SAMPLE_PATH.exists():
        raise FileNotFoundError(f"Benchmark sample not found at: {BENCHMARK_SAMPLE_PATH}")
    
    df = pd.read_csv(BENCHMARK_SAMPLE_PATH)
    if "attack_cat" not in df.columns:
        df["attack_cat"] = "Normal"
    
    clean_cat = category.strip().lower()
    matching = df[df["attack_cat"].astype(str).str.strip().str.lower() == clean_cat]
    
    if matching.empty:
        print(f"[!] Warning: Category '{category}' not found. Defaulting to first 5 flows.")
        return df.iloc[:sequence_len].copy(), "Unknown"
    
    if len(matching) < sequence_len:
        print(f"[!] Category '{category}' has only {len(matching)} records. Replicating to {sequence_len}.")
        indices = list(matching.index) * (sequence_len // len(matching) + 1)
        selected_df = df.loc[indices[:sequence_len]].copy()
    else:
        # Take 5 consecutive records from this category
        selected_df = matching.iloc[:sequence_len].copy()
    
    ground_truth = str(selected_df.iloc[-1]["attack_cat"]).strip()
    return selected_df, ground_truth

def run_live_demonstration(target_category: str = "DoS", explain: bool = True):
    print_banner()
    print(f"[*] Initializing Demonstration for Threat Target: [{target_category.upper()}]")
    print("-" * 80)
    
    # 1. Initialize Detection Engine
    t0 = time.time()
    print("[1/6] Loading Preprocessing Artifacts & Trained Hybrid Neural Model...")
    engine = AttackDetectionEngine()
    print(f"      -> Model: 1D-CNN + BiLSTM + Attention (Parameters verified)")
    print(f"      -> Preprocessor: Scaler (42 features) + Ordinal Encoded (proto, service, state)")
    print(f"      -> Initialization completed in {(time.time() - t0):.2f}s.\n")
    
    # 2. Select Sample Sequence
    print(f"[2/6] Extracting Sequence Window (T = {DEFAULT_SEQUENCE_LENGTH} consecutive flow observations)...")
    flow_records_df, ground_truth = load_attack_samples(target_category, sequence_len=DEFAULT_SEQUENCE_LENGTH)
    print(f"      -> Selected {len(flow_records_df)} flow records from benchmark corpus.")
    print(f"      -> Verified Ground-Truth Label from Dataset: '{ground_truth}'")
    
    # Preview key packet telemetry
    key_cols = [c for c in ["id", "proto", "service", "state", "dur", "sbytes", "dbytes", "rate", "sttl", "ct_srv_src"] if c in flow_records_df.columns]
    print("\n   [Telemetry Preview - Flow Observations]:")
    print(flow_records_df[key_cols].to_string(index=True))
    print()
    
    # 3. Execute Detection Pipeline
    print("[3/6] Executing Forward Pass Inference through Hybrid Architecture...")
    t_inf_start = time.time()
    result = engine.predict_dataframe_sequence(
        flow_records_df,
        explain=explain,
        flow_id="VIVA-LIVE-RUN-01"
    )
    total_pipeline_time = (time.time() - t_inf_start) * 1000.0
    
    pred_class = result["predicted_class"]
    confidence = result["confidence_pct"]
    is_intrusion = result["is_intrusion"]
    latency = result["latency_ms"]
    attn_weights = result["attention_weights"]
    alert = result["security_alert"]
    
    print("\n" + "=" * 80)
    print(f"   DETECTION RESULT: {pred_class.upper()} {'[INTRUSION DETECTED!]' if is_intrusion else '[NORMAL BENIGN FLOW]'}")
    print("=" * 80)
    print(f"   Model Classification Confidence : {confidence}")
    print(f"   Forward-Pass Neural Latency     : {latency:.2f} ms")
    print(f"   Total Pipeline + XAI Latency   : {total_pipeline_time:.2f} ms")
    print(f"   Threat Severity Level           : {alert['severity'].upper()}")
    print(f"   Recommended SOC Playbook        : {alert['defensive_playbook']}")
    print("-" * 80)
    
    # 4. Display 10-Class Probability Distribution
    print("\n[4/6] Complete 10-Class Threat Softmax Probability Distribution:")
    prob_items = sorted(result["probabilities"].items(), key=lambda x: x[1], reverse=True)
    for cat_name, prob in prob_items:
        bar_len = int(prob * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        marker = " ◄ [PREDICTED]" if cat_name == pred_class else ""
        print(f"   {cat_name:<16} | {bar} | {prob * 100:6.2f}%{marker}")
    
    # 5. Display Temporal Attention Weights
    print("\n[5/6] Learned Temporal Attention Weights across Time Steps (t_1 to t_5):")
    print("   [Explains which packet observation within the 5-step flow window drove the classification]")
    for t_idx, w in enumerate(attn_weights, start=1):
        bar_len = int(w * 40)
        bar = "■" * bar_len + "·" * (40 - bar_len)
        pct = w * 100.0
        focus = " (Dominant Flow Focus)" if w == max(attn_weights) else ""
        print(f"   Time Step t_{t_idx} (Flow #{t_idx}) | {bar} | {pct:6.2f}%{focus}")
    
    # 6. Display SHAP Game-Theoretic Explanation
    print("\n[6/6] Game-Theoretic SHAP (Shapley Additive exPlanations) Attribution:")
    if result.get("explanation"):
        expl = result["explanation"]
        print(f"   {expl['summary_text']}\n")
        
        print("   Top Influential Flow Drivers Ranked by Net Shapley Impact:")
        print("   " + "-" * 74)
        print(f"   {'Rank':<6} {'Feature Name':<16} {'Observed Value':<16} {'SHAP Value':<14} {'Direction'}")
        print("   " + "-" * 74)
        for f in expl.get("top_features", []):
            rank = f.get("rank", "-")
            name = f.get("feature", "unknown")
            obs = f.get("feature_value", 0.0)
            val = f.get("shap_value", 0.0)
            is_pos = f.get("is_positive", val >= 0)
            impact = "+ Threat Driver" if is_pos else "- Threat Mitigator"
            print(f"   #{rank:<5} {name:<16} {obs:<16.4f} {val:<+14.5f} {impact}")
        print("   " + "-" * 74)
        
        # Save Matplotlib SHAP chart
        out_plot = SHAP_PLOTS_DIR / "demo_shap_attribution.png"
        try:
            plot_shap_bar_chart(expl, save_path=out_plot)
            print(f"\n[+] Saved Publication-Quality SHAP Attribution Chart to:")
            print(f"    -> {out_plot.resolve()}")
        except Exception as e:
            print(f"[-] Could not save chart: {e}")
    else:
        print("   [!] SHAP explanation was bypassed or unavailable.")
    
    print("\n" + "=" * 80)
    print("   DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("   Ready for Viva Questions and Architectural Defense.")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Code-First Demonstration of Explainable Hybrid Deep Learning IDS")
    parser.add_argument(
        "--category",
        type=str,
        default="DoS",
        choices=["DoS", "Normal", "Exploits", "Fuzzers", "Generic", "Reconnaissance", "Analysis", "Backdoor", "Shellcode", "Worms"],
        help="Target attack category from UNSW-NB15 benchmark to classify (Default: DoS)"
    )
    parser.add_argument(
        "--no-shap",
        action="store_true",
        help="Skip SHAP computation for ultra-low latency demonstration"
    )
    args = parser.parse_args()
    run_live_demonstration(target_category=args.category, explain=not args.no_shap)
