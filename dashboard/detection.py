"""
Dashboard Page 4: Live Attack Detection Engine & Final Project Demonstration.
Supports selecting real UNSW-NB15 test traffic (DoS, Normal, Exploits, etc.) or uploading CSV flows,
running the hybrid model pipeline, displaying SOC intrusion alert banners, confidence gauges,
all 10-class probabilities, temporal attention distribution, and direct SHAP handoff.
"""

from pathlib import Path
import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from config.config import (
    TEST_DATASET_PATH,
    BENCHMARK_SAMPLE_PATH,
    LABEL_MAPPING,
    CLASS_NAMES,
    DEMO_ATTACK_CLASS,
    SEVERITY_LEVELS
)
from preprocessing.load_dataset import load_csv_data
from detection.predict_attack import AttackDetectionEngine
from explainability.feature_visualization import create_temporal_attention_chart, create_interactive_shap_bar_chart

@st.cache_resource
def get_detection_engine() -> AttackDetectionEngine:
    """Caches detection engine in memory for instant inference."""
    return AttackDetectionEngine()

def load_sample_library() -> pd.DataFrame:
    """Loads a representative cache of real UNSW-NB15 test traffic flows."""
    target_path = None
    if TEST_DATASET_PATH.exists():
        target_path = TEST_DATASET_PATH
    elif BENCHMARK_SAMPLE_PATH.exists():
        target_path = BENCHMARK_SAMPLE_PATH

    if target_path is None:
        return pd.DataFrame()

    df_test = load_csv_data(target_path)
    df_test.columns = [str(c).strip() for c in df_test.columns]

    if "attack_cat" not in df_test.columns:
        if "label" in df_test.columns:
            df_test["attack_cat"] = df_test["label"].map({0: "Normal", 1: "DoS"}).fillna("Normal")
        else:
            df_test["attack_cat"] = "Normal"

    df_test["attack_cat"] = df_test["attack_cat"].astype(str).str.strip()

    dfs = []
    for cat in df_test["attack_cat"].unique():
        sub = df_test[df_test["attack_cat"].str.lower() == cat.lower()].head(25)
        if not sub.empty:
            dfs.append(sub)

    if dfs:
        samples = pd.concat(dfs, ignore_index=True)
    else:
        samples = df_test.head(250).copy()

    return samples

def render_detection_page():
    st.markdown("""
    <div class="soc-header">
        <h1 class="soc-header-title">🎯 LIVE ATTACK DETECTION & INTRUSION INFERENCE</h1>
        <div class="soc-header-subtitle">
            Primary Demonstration Engine — Multiclass Network Traffic Classification with DoS Focus
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Check engine availability
    try:
        engine = get_detection_engine()
    except Exception as e:
        st.error(f"Detection engine failed to load: {e}. Please ensure the model is trained in the Model Training page.")
        return

    # 1. Traffic Input Selection
    st.markdown("### 📡 Step 1: Select or Upload Network Traffic")
    input_source = st.radio(
        "Choose Input Method:",
        ["Select Real Test Flow from UNSW-NB15 Benchmark", "Upload Custom Capture CSV (Live Flow)"],
        horizontal=True
    )

    selected_traffic_df = None
    ground_truth_label = None

    if input_source == "Select Real Test Flow from UNSW-NB15 Benchmark":
        samples_df = load_sample_library()
        if samples_df.empty:
            st.error("Test dataset not available in dataset/ directory.")
            return

        if "attack_cat" not in samples_df.columns:
            samples_df["attack_cat"] = "Normal"

        icol1, icol2 = st.columns([1, 2])
        with icol1:
            category_filter = st.selectbox(
                "Filter by True Ground-Truth Category:",
                ["DoS (Primary Demo)", "Normal", "Exploits", "Fuzzers", "Generic", "Reconnaissance", "Analysis", "Backdoor", "Shellcode", "Worms"]
            )
            cat_clean = "DoS" if "DoS" in category_filter else category_filter
            filtered_samples = samples_df[samples_df["attack_cat"].astype(str).str.strip().str.lower() == cat_clean.lower()]

        with icol2:
            if not filtered_samples.empty:
                sample_idx = st.selectbox(
                    f"Select Flow Index ({len(filtered_samples)} available in benchmark sample):",
                    options=list(filtered_samples.index),
                    format_func=lambda idx: f"Flow #{idx} | Service: {filtered_samples.loc[idx, 'service']} | Proto: {filtered_samples.loc[idx, 'proto']} | sbytes: {filtered_samples.loc[idx, 'sbytes']} | rate: {filtered_samples.loc[idx, 'rate']:.1f}"
                )
                # Create a 5-observation window starting at or around sample_idx
                start_i = max(0, sample_idx - 4)
                selected_traffic_df = samples_df.iloc[start_i:sample_idx + 1].copy()
                ground_truth_label = str(samples_df.loc[sample_idx, "attack_cat"]).strip()
            else:
                st.warning(f"No samples found for category: {cat_clean}")

    else:
        uploaded_file = st.file_uploader("Upload Network Flow CSV (1 to N rows):", type=["csv"])
        if uploaded_file is not None:
            selected_traffic_df = pd.read_csv(uploaded_file, low_memory=False)
            if "attack_cat" in selected_traffic_df.columns:
                ground_truth_label = str(selected_traffic_df.iloc[-1]["attack_cat"]).strip()
        else:
            st.info("Upload a CSV file containing network packet flow records.")

    # 2. Preview Selected Flow Features
    if selected_traffic_df is not None:
        st.markdown("### 🔍 Step 2: Selected Traffic Flow Telemetry")
        if ground_truth_label:
            st.markdown(f"**Ground-Truth Label from Dataset:** `{ground_truth_label}` (Used strictly to verify detection correctness)")

        # Display important flow features
        key_cols = [c for c in ["id", "proto", "service", "state", "dur", "sbytes", "dbytes", "rate", "sttl", "ct_srv_src", "attack_cat"] if c in selected_traffic_df.columns]
        st.dataframe(selected_traffic_df[key_cols].tail(5), use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        # 3. Detection Action Button
        run_detection = st.button("⚡ EXECUTE HYBRID DEEP LEARNING DETECTION PIPELINE", use_container_width=True)

        if run_detection:
            with st.spinner("Processing through 1D-CNN + BiLSTM + Attention Hybrid Architecture..."):
                t_start = time.time()
                # Run prediction
                res = engine.predict_dataframe_sequence(
                    selected_traffic_df,
                    explain=True, # Automatically compute genuine SHAP
                    flow_id=f"FLOW-{int(time.time()*1000)%100000}"
                )
                pred_latency = (time.time() - t_start) * 1000.0

                # Cache in session state for explainability page handoff
                st.session_state["latest_detection_result"] = res
                st.session_state["latest_traffic_df"] = selected_traffic_df

        # If a detection result exists in session state, display it
        if "latest_detection_result" in st.session_state:
            res = st.session_state["latest_detection_result"]
            pred_class = res["predicted_class"]
            confidence = res["confidence"]
            is_intrusion = res["is_intrusion"]
            alert = res["security_alert"]

            st.markdown("---")
            st.markdown("### 🚨 Step 3: PREDICTION & THREAT TELEMETRY RESULT")

            # Alert Banner
            if is_intrusion:
                st.markdown(f"""
                <div class="alert-banner-critical">
                    <div style="font-size: 22px; font-weight: 800; color: #EF4444; display: flex; align-items: center; gap: 10px;">
                        ⚠️ INTRUSION DETECTED: {pred_class.upper()} ATTACK
                        <span class="badge-critical" style="margin-left: auto;">{alert['severity']} SEVERITY</span>
                    </div>
                    <div style="margin-top: 8px; font-size: 14px; color: #FCA5A5;">
                        <b>Detection Confidence:</b> {res['confidence_pct']} | 
                        <b>Latency:</b> {res['latency_ms']} ms | 
                        <b>Architecture:</b> {res['model_name']}
                    </div>
                    <div style="margin-top: 10px; font-size: 13px; color: #E5E7EB; background: rgba(0,0,0,0.3); padding: 10px; border-radius: 6px;">
                        <b>SOC Incident Response Playbook:</b> {alert['defensive_playbook']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="alert-banner-normal">
                    <div style="font-size: 22px; font-weight: 800; color: #10B981; display: flex; align-items: center; gap: 10px;">
                        ✅ BENIGN TRAFFIC CONFIRMED: NORMAL FLOW
                        <span class="badge-normal" style="margin-left: auto;">INFORMATIONAL</span>
                    </div>
                    <div style="margin-top: 8px; font-size: 14px; color: #A7F3D0;">
                        <b>Confidence:</b> {res['confidence_pct']} | 
                        <b>Latency:</b> {res['latency_ms']} ms | 
                        <b>Status:</b> All packet connection signatures conform to standard protocols.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Probabilities Chart and Attention Weights
            pcol1, pcol2 = st.columns([1.2, 0.8])
            with pcol1:
                st.markdown("#### Complete 10-Class Probability Distribution")
                prob_df = pd.DataFrame(list(res["probabilities"].items()), columns=["Category", "Probability"])
                
                fig_prob = px.bar(
                    prob_df,
                    x="Probability",
                    y="Category",
                    orientation="h",
                    text_auto=".1%",
                    color="Category",
                    color_discrete_map={
                        pred_class: "#EF4444" if is_intrusion else "#10B981",
                        "Normal": "#10B981" if pred_class != "Normal" else "#10B981"
                    }
                )
                fig_prob.update_layout(
                    paper_bgcolor="#111827",
                    plot_bgcolor="#1F2937",
                    font=dict(color="#E5E7EB"),
                    xaxis=dict(range=[0, 1.05], title="Model Softmax Probability"),
                    yaxis=dict(title="Attack Category"),
                    showlegend=False,
                    height=350,
                    margin=dict(l=60, r=30, t=20, b=30)
                )
                st.plotly_chart(fig_prob, use_container_width=True)

            with pcol2:
                st.markdown("#### Observation Temporal Attention Weights")
                if "attention_weights" in res and res["attention_weights"]:
                    attn_chart = create_temporal_attention_chart(
                        np.array(res["attention_weights"]),
                        sequence_length=res["sequence_length"]
                    )
                    st.plotly_chart(attn_chart, use_container_width=True)

            # Quick SHAP Summary & Jump to Explainability Button
            if res.get("explanation"):
                st.markdown("---")
                st.markdown("### 🧠 Step 4: Explainable AI Quick Brief")
                expl = res["explanation"]
                if isinstance(expl, dict) and "summary_text" in expl:
                    st.markdown(expl["summary_text"])

                if st.button("🔬 Open Dedicated Explainability Page ➔", use_container_width=True):
                    st.session_state["nav_page"] = "🔬 Explainable AI (SHAP)"
                    st.session_state["sidebar_nav_radio"] = "🔬 Explainable AI (SHAP)"
                    st.rerun()

                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("📊 View Detailed SHAP Feature Dynamics Inline", expanded=True):
                    try:
                        st.markdown("#### Positive & Negative Feature Attributions")
                        shap_fig = create_interactive_shap_bar_chart(expl)
                        st.plotly_chart(shap_fig, use_container_width=True)

                        # Top features table
                        top_f = expl.get("top_features", []) if isinstance(expl, dict) else []
                        if top_f and isinstance(top_f, list):
                            st.markdown("#### Top Influential Network Flow Drivers")
                            top_rows = []
                            for rank, item in enumerate(top_f, 1):
                                if not isinstance(item, dict):
                                    continue
                                is_pos = item.get("is_positive", item.get("shap_value", 0) >= 0)
                                impact_str = "🔺 Threat Contributor" if is_pos else "🛡️ Threat Mitigator"
                                obs_val = item.get("feature_value", item.get("observed_value", 0.0))
                                shap_val = item.get("shap_value", 0.0)
                                top_rows.append({
                                    "Rank": f"#{rank}",
                                    "Feature": item.get("feature", "Unknown"),
                                    "Observed Value": f"{float(obs_val):.4f}",
                                    "SHAP Value": f"{float(shap_val):+.5f}",
                                    "Impact": impact_str,
                                    "Domain Cyber Meaning": item.get("cyber_context", "")
                                })
                            if top_rows:
                                st.dataframe(pd.DataFrame(top_rows), use_container_width=True, hide_index=True)
                    except Exception as e:
                        st.warning(f"Note: Inline SHAP dynamic view could not be rendered: {e}")
                        st.info("Full interactive SHAP attribution and cyber interpretations can be viewed on the dedicated Explainability page.")

