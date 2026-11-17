"""
Main Streamlit Application Entrypoint for Explainable Hybrid Deep Learning IDS.
Coordinates multi-page SOC dashboard navigation, state persistence, and styling.
"""

import sys
from pathlib import Path

# Add project root to path
_app_dir = Path(__file__).resolve().parent
if str(_app_dir) not in sys.path:
    sys.path.insert(0, str(_app_dir))

import os
os.environ["KERAS_BACKEND"] = "torch"
import streamlit as st

# Configure Page
st.set_page_config(
    page_title="Explainable Hybrid Deep Learning IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

from dashboard.styles import apply_soc_theme
from dashboard.overview import render_overview_page
from dashboard.dataset_explorer import render_dataset_explorer_page
from dashboard.training import render_training_page
from dashboard.detection import render_detection_page
from dashboard.explainability import render_explainability_page
from dashboard.performance import render_performance_page
from dashboard.alerts import render_alerts_page
from dashboard.system_info import render_system_info_page
from config.config import MODELS_DIR, PREPROCESSORS_DIR

# Apply Custom SOC Aesthetic
apply_soc_theme()

# Navigation Pages
PAGES = {
    "🛡️ SOC Command Overview": render_overview_page,
    "📁 Dataset Explorer": render_dataset_explorer_page,
    "⚡ Model Training": render_training_page,
    "🎯 Attack Detection (Demo)": render_detection_page,
    "🔬 Explainable AI (SHAP)": render_explainability_page,
    "📊 Model Performance": render_performance_page,
    "🚨 Incident History": render_alerts_page,
    "ℹ️ System Architecture": render_system_info_page
}

# Sidebar Navigation
st.sidebar.markdown("""
<div style="text-align: center; padding: 12px 0 18px 0; border-bottom: 1px solid #1E293B;">
    <div style="font-size: 26px;">🛡️</div>
    <div style="font-weight: 800; font-size: 16px; color: #F8FAFC; letter-spacing: -0.3px; margin-top: 4px;">
        HYBRID DEEP IDS
    </div>
    <div style="font-size: 11px; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.8px;">
        Explainable Network Security
    </div>
</div>
""", unsafe_allow_html=True)

# System Health Pill in Sidebar
model_ready = (MODELS_DIR / "hybrid_ids_model.keras").exists() and (PREPROCESSORS_DIR / "preprocessing_pipeline.joblib").exists()
status_color = "#10B981" if model_ready else "#F59E0B"
status_text = "MODEL READY" if model_ready else "UNTRAINED"

st.sidebar.markdown(f"""
<div style="background-color: #111827; border: 1px solid #1F2937; border-radius: 6px; padding: 10px 14px; margin: 16px 0; display: flex; align-items: center; justify-content: space-between;">
    <span style="font-size: 12px; color: #9CA3AF; font-weight: 600;">ENGINE STATE:</span>
    <span style="font-size: 11px; font-weight: 700; color: {status_color}; background: rgba(0,0,0,0.3); padding: 2px 8px; border-radius: 4px;">
        ● {status_text}
    </span>
</div>
""", unsafe_allow_html=True)

# Synchronize Navigation State
nav_keys = list(PAGES.keys())

# If another page requested a programmatic page navigation
if "nav_page" in st.session_state:
    target = st.session_state["nav_page"]
    for k in nav_keys:
        if target.lower() in k.lower():
            st.session_state["sidebar_nav_radio"] = k
            break
    del st.session_state["nav_page"]

if "sidebar_nav_radio" not in st.session_state:
    st.session_state["sidebar_nav_radio"] = nav_keys[0]

selected_page = st.sidebar.radio(
    "Navigation Console:",
    nav_keys,
    key="sidebar_nav_radio"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size: 11px; color: #64748B; line-height: 1.5;">
    <b>Research Contribution:</b><br>
    Explainable Multiclass Intrusion Detection via 1D-CNN, Bidirectional LSTM, and Temporal Attention with Game-Theoretic SHAP Attributions.
    <br><br>
    <b>Primary Target:</b> Denial of Service (DoS)<br>
    <b>Dataset:</b> UNSW-NB15 Benchmark
</div>
""", unsafe_allow_html=True)

# Render Target Page
page_fn = PAGES[selected_page]
page_fn()

