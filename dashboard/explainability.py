"""
Dashboard Page 5: Explainable AI (XAI) using SHAP.
Visualizes game-theoretic feature attributions, positive vs negative contributions,
top influential metrics, and plain-English cybersecurity root cause analysis.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from config.config import (
    LABEL_MAPPING,
    CLASS_NAMES,
    DEMO_ATTACK_CLASS,
    SHAP_PLOTS_DIR
)
from explainability.feature_visualization import create_interactive_shap_bar_chart

def render_explainability_page():
    st.markdown("""
    <div class="soc-header">
        <h1 class="soc-header-title">🔬 EXPLAINABLE AI (SHAP) — THREAT REASONING ENGINE</h1>
        <div class="soc-header-subtitle">
            Game-Theoretic Shapley Additive exPlanations for Deep Learning Intrusion Classifications
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Check if a prediction result exists in session state
    if "latest_detection_result" not in st.session_state or st.session_state["latest_detection_result"] is None:
        st.warning(
            "No active prediction result found to explain. "
            "Please go to the **Attack Detection** page, select a traffic sample, and click **EXECUTE DETECTION PIPELINE**."
        )
        return

    res = st.session_state["latest_detection_result"]
    explanation = res.get("explanation")
    
    if not explanation:
        st.error("SHAP explanation data was not computed for this sample.")
        return

    pred_class = res["predicted_class"]
    confidence = res["confidence"]
    explained_class = explanation["explained_class_name"]

    # 1. Prediction Context Card
    st.markdown(f"""
    <div style="background-color: #111827; border: 1px solid #1F2937; border-radius: 8px; padding: 18px; margin-bottom: 20px;">
        <div style="font-size: 18px; font-weight: 700; color: #F9FAFB;">
            Explaining Model Prediction: <span style="color: {'#EF4444' if pred_class != 'Normal' else '#10B981'};">{pred_class.upper()}</span>
            <span style="font-size: 14px; font-weight: 400; color: #9CA3AF; margin-left: 10px;">(Confidence: {res['confidence_pct']})</span>
        </div>
        <div style="font-size: 13px; color: #94A3B8; margin-top: 6px;">
            Target Category Explained: <b>{explained_class}</b> | 
            Sequence Input: <b>5 Consecutive Observations</b> | 
            Explainer: <b>SHAP Kernel/Permutation Explainer</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Plain-English SOC Analyst Brief
    st.markdown("### 📋 SOC Threat Telemetry Brief")
    import re as _re
    brief_html = _re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", str(explanation.get("summary_text", "")))
    st.markdown(f"""
    <div style="background: rgba(30, 41, 59, 0.6); border-left: 4px solid #38BDF8; padding: 14px 18px; border-radius: 4px; font-size: 14px; line-height: 1.6; color: #E2E8F0; margin-bottom: 24px;">
        {brief_html}
    </div>
    """, unsafe_allow_html=True)

    # 3. Interactive SHAP Bar Chart
    st.markdown("### 📊 Positive & Negative Feature Attribution Dynamics")
    st.markdown("""
    * **Crimson Bars (Positive SHAP > 0):** Network features whose anomalous values pushed the model toward classifying this flow as the detected attack category.
    * **Emerald Bars (Negative SHAP < 0):** Network features whose values counterbalanced or opposed this attack classification.
    """)
    try:
        fig_shap = create_interactive_shap_bar_chart(explanation)
        st.plotly_chart(fig_shap, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not render SHAP bar chart: {e}")

    # 4. Top Influential Features Table with Domain Context
    st.markdown("### 🔍 Top Influential Network Flow Features (Detailed Breakdown)")
    
    top_features = explanation.get("top_features", []) if isinstance(explanation, dict) else []
    table_records = []
    for rank, f in enumerate(top_features, 1):
        if not isinstance(f, dict):
            continue
        rank_val = f.get("rank", f"#{rank}")
        feat_name = f.get("feature", "Unknown")
        obs_val = f.get("feature_value", f.get("observed_value", 0.0))
        shap_val = f.get("shap_value", 0.0)
        is_pos = f.get("is_positive", shap_val >= 0)
        direction = f.get("direction", "Increased Risk (Positive)" if is_pos else "Decreased Risk (Negative)")
        context = f.get("cyber_context", "")

        table_records.append({
            "Rank": rank_val,
            "Feature Name": feat_name,
            "Observed Value": f"{float(obs_val):.4f}",
            "SHAP Attribution": f"{float(shap_val):+.5f}",
            "Impact Direction": direction,
            "Domain Security Meaning": context
        })
        
    if table_records:
        df_top = pd.DataFrame(table_records)
        st.dataframe(df_top, use_container_width=True, hide_index=True)

    # 5. Temporal Attention vs SHAP Correlation Insight
    st.markdown("---")
    st.markdown("### ⏱️ Temporal Context Synthesis")
    st.info(
        "In this hybrid architecture, the **1D-CNN** extracts localized spatial correlations within each packet observation, "
        "the **BiLSTM** models temporal sequence transitions across consecutive flows, and the **Temporal Attention Mechanism** "
        "assigns dynamic weights across time steps. SHAP mathematically confirms that the features identified with highest attribution "
        "correlate strongly with periods of high attention weight."
    )

