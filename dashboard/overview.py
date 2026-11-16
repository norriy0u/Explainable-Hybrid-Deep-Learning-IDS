"""
Dashboard Page 1: Security Operations Center (SOC) Command Center Overview.
Displays high-level KPI cards, model operational status, threat breakdown, and live alert feeds.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config.config import (
    MODELS_DIR,
    PREPROCESSORS_DIR,
    REPORTS_DIR,
    SEVERITY_COLORS,
    LABEL_MAPPING,
    DEMO_ATTACK_CLASS
)
from utils.helpers import load_json
from detection.detection_history import DetectionHistoryManager

def render_overview_page():
    st.markdown("""
    <div class="soc-header">
        <h1 class="soc-header-title">🛡️ SOC COMMAND CENTER — HYBRID DEEP LEARNING IDS</h1>
        <div class="soc-header-subtitle">
            Autonomous Network Intrusion Detection & Explainable AI Threat Telemetry System
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 1. System Status Indicators
    model_path = MODELS_DIR / "hybrid_ids_model.keras"
    pipeline_path = PREPROCESSORS_DIR / "preprocessing_pipeline.joblib"
    eval_path = REPORTS_DIR / "evaluation_results.json"
    
    is_model_trained = model_path.exists() and pipeline_path.exists()
    eval_data = load_json(eval_path) if eval_path.exists() else {}
    history_mgr = DetectionHistoryManager()
    alerts = history_mgr.get_all()
    
    total_alerts = len(alerts)
    intrusion_alerts = len([a for a in alerts if a.get("is_intrusion", False)])
    normal_flows = total_alerts - intrusion_alerts
    dos_alerts = len([a for a in alerts if a.get("predicted_class") == DEMO_ATTACK_CLASS])

    # 2. Top KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">System Status</div>
            <div class="kpi-value" style="color: {'#10B981' if is_model_trained else '#F59E0B'}; font-size: 20px;">
                {'● OPERATIONAL' if is_model_trained else '○ UNTRAINED'}
            </div>
            <div class="kpi-delta" style="color: #94A3B8;">{'Model Loaded' if is_model_trained else 'Action Required'}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        accuracy_display = f"{eval_data.get('accuracy', 0.0) * 100:.1f}%" if eval_data else "N/A"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Test Accuracy</div>
            <div class="kpi-value" style="color: #38BDF8;">{accuracy_display}</div>
            <div class="kpi-delta" style="color: #10B981;">Macro F1: {eval_data.get('macro_f1', 0)*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Flows Evaluated</div>
            <div class="kpi-value" style="color: #F8FAFC;">{total_alerts:,}</div>
            <div class="kpi-delta" style="color: #94A3B8;">Active session log</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Intrusions Flagged</div>
            <div class="kpi-value" style="color: #EF4444;">{intrusion_alerts:,}</div>
            <div class="kpi-delta" style="color: #F87171;">Normal: {normal_flows:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">DoS Attacks Detected</div>
            <div class="kpi-value" style="color: #F97316;">{dos_alerts:,}</div>
            <div class="kpi-delta" style="color: #FB923C;">Primary Demo Focus</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Two-Column Visualizations: Threat Breakdown & Architecture Specs
    col_left, col_right = st.columns([1.1, 0.9])

    with col_left:
        st.markdown("### 📊 Attack Category Telemetry Breakdown")
        if alerts:
            df_alerts = pd.DataFrame(alerts)
            cat_counts = df_alerts["predicted_class"].value_counts().reset_index()
            cat_counts.columns = ["Attack Category", "Count"]

            fig_donut = px.pie(
                cat_counts,
                names="Attack Category",
                values="Count",
                hole=0.45,
                color="Attack Category",
                color_discrete_map={
                    "Normal": "#10B981",
                    "DoS": "#EF4444",
                    "Exploits": "#F97316",
                    "Fuzzers": "#F59E0B",
                    "Generic": "#8B5CF6",
                    "Reconnaissance": "#06B6D4",
                    "Analysis": "#3B82F6",
                    "Backdoor": "#EC4899",
                    "Shellcode": "#DC2626",
                    "Worms": "#B91C1C"
                }
            )
            fig_donut.update_layout(
                paper_bgcolor="#111827",
                plot_bgcolor="#111827",
                font=dict(color="#E5E7EB"),
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_donut, use_container_width=True)
        else:
            st.info("No network flow telemetry recorded yet. Run a live detection on the **Attack Detection** page to populate.")

    with col_right:
        st.markdown("### ⚙️ Engine Specifications & Status")
        st.markdown("""
        <div style="background-color: #111827; border: 1px solid #1F2937; border-radius: 8px; padding: 18px;">
            <table style="width: 100%; font-size: 13px; color: #E5E7EB; border-collapse: collapse;">
                <tr style="border-bottom: 1px solid #1F2937; height: 34px;">
                    <td style="color: #9CA3AF;"><b>Deep Learning Core:</b></td>
                    <td style="font-family: monospace; color: #38BDF8;">CNN + BiLSTM + Attention</td>
                </tr>
                <tr style="border-bottom: 1px solid #1F2937; height: 34px;">
                    <td style="color: #9CA3AF;"><b>Observation Window (T):</b></td>
                    <td style="font-family: monospace;">5 Consecutive Packets/Flows</td>
                </tr>
                <tr style="border-bottom: 1px solid #1F2937; height: 34px;">
                    <td style="color: #9CA3AF;"><b>Input Dimensions:</b></td>
                    <td style="font-family: monospace;">(Batch, 5, 42 Features)</td>
                </tr>
                <tr style="border-bottom: 1px solid #1F2937; height: 34px;">
                    <td style="color: #9CA3AF;"><b>Classification Head:</b></td>
                    <td style="font-family: monospace;">10-Class Softmax</td>
                </tr>
                <tr style="border-bottom: 1px solid #1F2937; height: 34px;">
                    <td style="color: #9CA3AF;"><b>Explainability Engine:</b></td>
                    <td style="font-family: monospace; color: #10B981;">SHAP Kernel / Permutation</td>
                </tr>
                <tr style="border-bottom: 1px solid #1F2937; height: 34px;">
                    <td style="color: #9CA3AF;"><b>Dataset Benchmark:</b></td>
                    <td style="font-family: monospace;">UNSW-NB15 Official Splits</td>
                </tr>
                <tr style="height: 34px;">
                    <td style="color: #9CA3AF;"><b>Primary Target Demo:</b></td>
                    <td style="font-family: monospace; color: #EF4444; font-weight: bold;">Denial of Service (DoS)</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    # 4. Recent Security Alert Feed
    st.markdown("### 🚨 Live Security Incident Feed")
    if alerts:
        recent_alerts = alerts[:8]
        display_records = []
        for a in recent_alerts:
            display_records.append({
                "Alert ID": a["alert_id"],
                "Timestamp": a["timestamp"],
                "Threat Category": a["predicted_class"],
                "Confidence": a["confidence_pct"],
                "Severity": a["severity"],
                "Playbook Action": a["defensive_playbook"][:65] + "..."
            })
        st.dataframe(pd.DataFrame(display_records), use_container_width=True)
    else:
        st.markdown("""
        <div style="background-color: #111827; border: 1px dashed #374151; border-radius: 8px; padding: 24px; text-align: center; color: #9CA3AF;">
            Active alert stream is currently quiet. Navigate to <b>Attack Detection</b> to test network traffic samples.
        </div>
        """, unsafe_allow_html=True)

