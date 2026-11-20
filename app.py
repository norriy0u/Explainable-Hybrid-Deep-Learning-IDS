"""
Main Streamlit Application Entrypoint for Explainable Hybrid Deep Learning IDS.
Coordinates multi-page SOC dashboard navigation, state persistence, safe lazy imports, and styling.
"""

import os
import sys
from pathlib import Path

# Configure Keras backend BEFORE any deep learning imports
os.environ.setdefault("KERAS_BACKEND", "torch")

# Ensure project root is on sys.path
_app_dir = Path(__file__).resolve().parent
if str(_app_dir) not in sys.path:
    sys.path.insert(0, str(_app_dir))

import importlib
import streamlit as st

# Configure Page
st.set_page_config(
    page_title="Explainable Hybrid Deep Learning IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

from dashboard.styles import apply_soc_theme
from config.config import MODELS_DIR, PREPROCESSORS_DIR

# Apply Custom SOC Aesthetic
apply_soc_theme()

# Lazy Page Registry: (module_path, render_function_name)
# Prevents non-ML pages (Overview, Dataset Explorer, Performance, Alerts) from importing
# Keras/PyTorch/SHAP at startup, speeding up initial load and isolating environment issues.
PAGES = {
    "🛡️ SOC Command Overview": ("dashboard.overview", "render_overview_page"),
    "📁 Dataset Explorer": ("dashboard.dataset_explorer", "render_dataset_explorer_page"),
    "⚡ Model Training": ("dashboard.training", "render_training_page"),
    "🎯 Attack Detection (Demo)": ("dashboard.detection", "render_detection_page"),
    "🔬 Explainable AI (SHAP)": ("dashboard.explainability", "render_explainability_page"),
    "📊 Model Performance": ("dashboard.performance", "render_performance_page"),
    "🚨 Incident History": ("dashboard.alerts", "render_alerts_page"),
    "ℹ️ System Architecture": ("dashboard.system_info", "render_system_info_page"),
}

# Sidebar Branding
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

if "current_page_name" not in st.session_state:
    st.session_state["current_page_name"] = nav_keys[0]

if "nav_page" in st.session_state:
    target = st.session_state["nav_page"]
    for k in nav_keys:
        if target.lower() in k.lower():
            st.session_state["current_page_name"] = k
            break
    del st.session_state["nav_page"]

current_idx = nav_keys.index(st.session_state["current_page_name"]) if st.session_state["current_page_name"] in nav_keys else 0

selected_page = st.sidebar.radio(
    "Navigation Console:",
    nav_keys,
    index=current_idx
)
st.session_state["current_page_name"] = selected_page

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

# Lazily import and render the selected page
module_name, func_name = PAGES[selected_page]
try:
    page_module = importlib.import_module(module_name)
    page_fn = getattr(page_module, func_name)
    page_fn()
except ImportError as exc:
    st.error(f"Could not load **{selected_page}** due to a Python dependency/DLL error: `{exc}`")
    if "Application Control" in str(exc) or "_ml_dtypes_ext" in str(exc) or "WinError 4551" in str(exc):
        st.warning(
            "**Windows Application Control (Smart App Control) Blocked a Native Python Extension**\n\n"
            "Your current virtual environment contains a native `.pyd`/`.dll` extension (such as `ml_dtypes` or `shap`) "
            "that Windows Application Control blocked from loading.\n\n"
            "**Safe Fix (No Security Changes Needed):**\n"
            "Launch the dashboard using the verified Python 3.14 interpreter (`C:\\Python314\\python.exe`) via `run_app.bat` or:\n"
            "```powershell\n"
            "C:\\Python314\\python.exe -m streamlit run app.py\n"
            "```"
        )
    else:
        st.info("Install project dependencies with `pip install -r requirements.txt` and restart Streamlit.")
