"""
Dashboard Page 6: Model Performance and Academic Evaluation Metrics.
Displays Confusion Matrix, Per-Class Precision/Recall/F1, Latency, Ablation Studies, and Sequence Length Comparisons.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from config.config import (
    REPORTS_DIR,
    PLOTS_DIR,
    EXPERIMENT_RESULTS_DIR,
    CLASS_NAMES,
    LABEL_MAPPING,
    DEMO_ATTACK_CLASS
)
from utils.helpers import load_json

def render_performance_page():
    st.markdown("""
    <div class="soc-header">
        <h1 class="soc-header-title">📊 MODEL PERFORMANCE & ABLATION BENCHMARKS</h1>
        <div class="soc-header-subtitle">
            Rigorous Academic Evaluation, Per-Class Classification Diagnostics, and Architectural Ablations
        </div>
    </div>
    """, unsafe_allow_html=True)

    eval_path = REPORTS_DIR / "evaluation_results.json"
    cm_path = PLOTS_DIR / "confusion_matrix.png"
    ablation_path = EXPERIMENT_RESULTS_DIR / "ablation_study.json"
    ablation_plot = PLOTS_DIR / "model_comparison.png"
    seq_path = EXPERIMENT_RESULTS_DIR / "sequence_length_study.json"
    seq_plot = PLOTS_DIR / "sequence_length_comparison.png"

    if not eval_path.exists():
        st.warning("Model evaluation metrics not found. Train the model first on the **Model Training** page.")
        return

    eval_data = load_json(eval_path)

    # 1. Summary Metric Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Overall Accuracy", f"{eval_data['accuracy']*100:.2f}%")
    col2.metric("Macro F1-Score", f"{eval_data['macro_f1']*100:.2f}%")
    col3.metric("Weighted F1-Score", f"{eval_data['weighted_f1']*100:.2f}%")
    
    dos_perf = eval_data.get("dos_performance", {})
    dos_f1 = dos_perf.get("f1_score", 0.0)
    col4.metric("DoS Attack F1", f"{dos_f1*100:.2f}%")
    
    avg_latency = eval_data.get("inference_latency_ms_per_sample")
    lat_str = f"{avg_latency:.3f} ms" if avg_latency else "< 1 ms"
    col5.metric("Avg Flow Latency", lat_str)

    st.markdown("---")

    # 2. Confusion Matrix & Per-Class Table
    pcol_left, pcol_right = st.columns([1.1, 0.9])

    with pcol_left:
        st.markdown("### 🧩 Confusion Matrix Heatmap")
        if cm_path.exists():
            st.image(str(cm_path), width="stretch", caption="UNSW-NB15 10-Class Confusion Matrix")
        else:
            cm = np.array(eval_data.get("confusion_matrix", []))
            if len(cm) > 0:
                fig_cm = px.imshow(
                    cm,
                    labels=dict(x="Predicted Class", y="True Class", color="Count"),
                    x=CLASS_NAMES,
                    y=CLASS_NAMES,
                    text_auto=True,
                    color_continuous_scale="Blues"
                )
                fig_cm.update_layout(paper_bgcolor="#111827", font=dict(color="#E5E7EB"))
                st.plotly_chart(fig_cm, use_container_width=True)

    with pcol_right:
        st.markdown("### 📋 Per-Class Classification Report")
        per_class = eval_data.get("per_class_metrics", {})
        table_rows = []
        for c_name, m in per_class.items():
            table_rows.append({
                "Category": c_name,
                "Precision": f"{m['precision']*100:.2f}%",
                "Recall": f"{m['recall']*100:.2f}%",
                "F1-Score": f"{m['f1_score']*100:.2f}%",
                "Support": f"{m['support']:,}"
            })
        df_pc = pd.DataFrame(table_rows)
        st.dataframe(df_pc, use_container_width=True, hide_index=True)

    # 3. Ablation Study Section
    st.markdown("---")
    st.markdown("### 🔬 Architectural Component Ablation Study")
    st.markdown("""
    To validate the research contribution of each individual module, we benchmark:
    1. **CNN Baseline:** Local feature extraction alone.
    2. **BiLSTM Baseline:** Bidirectional sequential modeling alone.
    3. **CNN + BiLSTM:** Serial combination without attention.
    4. **CNN + BiLSTM + Attention:** Hybrid architecture without class weighting.
    5. **Full Proposed Model:** CNN + BiLSTM + Attention + Balanced Class Weighting.
    """)

    if ablation_path.exists():
        ab_data = load_json(ablation_path)
        ab_records = []
        for m_name, m_metrics in ab_data.items():
            dos_m = m_metrics.get("dos_performance", {})
            ab_records.append({
                "Architecture": m_name,
                "Accuracy": f"{m_metrics['accuracy']*100:.2f}%",
                "Macro F1": f"{m_metrics['macro_f1']*100:.2f}%",
                "Weighted F1": f"{m_metrics['weighted_f1']*100:.2f}%",
                "DoS F1": f"{dos_m.get('f1_score', 0)*100:.2f}%",
                "Train Time (s)": f"{m_metrics.get('training_time_s', 'N/A')}s"
            })
        st.dataframe(pd.DataFrame(ab_records), use_container_width=True, hide_index=True)
        
        if ablation_plot.exists():
            st.image(str(ablation_plot), width="stretch", caption="Comparative Performance Across Model Ablations")
    else:
        st.info("Ablation study is currently computing or has not been triggered yet.")

    # 4. Sequence Length Ablation (T = 3, 5, 10)
    st.markdown("---")
    st.markdown("### ⏱️ Sequence Length Window Ablation (T = 3, 5, 10)")
    if seq_path.exists():
        seq_data = load_json(seq_path)
        seq_records = []
        for t_name, t_metrics in seq_data.items():
            dos_t = t_metrics.get("dos_performance", {})
            seq_records.append({
                "Window Length (T)": t_name,
                "Accuracy": f"{t_metrics['accuracy']*100:.2f}%",
                "Macro F1": f"{t_metrics['macro_f1']*100:.2f}%",
                "Weighted F1": f"{t_metrics['weighted_f1']*100:.2f}%",
                "DoS F1": f"{dos_t.get('f1_score', 0)*100:.2f}%"
            })
        st.dataframe(pd.DataFrame(seq_records), use_container_width=True, hide_index=True)
        if seq_plot.exists():
            st.image(str(seq_plot), width="stretch", caption="Impact of Observation Sequence Length (T) on Intrusion Detection")
    else:
        st.info("Sequence length comparison results will appear once the ablation study finishes.")

