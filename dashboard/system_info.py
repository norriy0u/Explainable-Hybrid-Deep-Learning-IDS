"""
Dashboard Page 8: System Architecture, Environment Specifications, and Research Integrity.
Documents the deep learning dataflow, mathematical foundations, and defensive cybersecurity policy.
"""

import sys
import platform
import streamlit as st
import pandas as pd
import keras
import torch
import shap
import sklearn

def render_system_info_page():
    st.markdown("""
    <div class="soc-header">
        <h1 class="soc-header-title">ℹ️ SYSTEM ARCHITECTURE & RESEARCH BLUEPRINT</h1>
        <div class="soc-header-subtitle">
            Deep Learning Mathematical Formulations, Preprocessing Specifications, and Environment Manifest
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Architectural Blueprint
    st.markdown("### 🏛️ Proposed Hybrid Deep Learning Architecture")
    st.markdown("""
    ```
    ┌────────────────────────────────────────────────────────────────────────┐
    │  INPUT: Network Observation Sequence (Batch_Size, T=5, D=42 Features)   │
    └───────────────────────────────────┬────────────────────────────────────┘
                                        │
                                        ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │  MODULE A: 1D-CNN (Temporal Spatial Extractor)                         │
    │  • Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')│
    │  • BatchNormalization + Dropout(0.25)                                  │
    │  • Output Shape: (Batch, 5, 64)                                        │
    └───────────────────────────────────┬────────────────────────────────────┘
                                        │
                                        ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │  MODULE B: BIDIRECTIONAL LSTM (Temporal Sequential Context)            │
    │  • Bidirectional(LSTM(units=64, return_sequences=True, dropout=0.20))  │
    │  • Forward LSTM (64) + Backward LSTM (64) -> (Batch, 5, 128)           │
    └───────────────────────────────────┬────────────────────────────────────┘
                                        │
                                        ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │  MODULE C: TRAINABLE TEMPORAL ATTENTION MECHANISM                      │
    │  • Alignment Score: e_t = tanh(h_t · W + b) · v                        │
    │  • Attention Weight: α_t = exp(e_t) / Σ_j exp(e_j)                     │
    │  • Context Vector: C = Σ_t (α_t · h_t) -> (Batch, 128)                 │
    └───────────────────────────────────┬────────────────────────────────────┘
                                        │
                                        ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │  MODULE D: CLASSIFICATION HEAD                                         │
    │  • Dense(64, activation='relu') + BatchNormalization + Dropout(0.25)   │
    │  • Dense(10, activation='softmax')                                     │
    │  • Output: [P(Normal), P(Analysis), P(Backdoor), P(DoS), ..., P(Worms)]│
    └───────────────────────────────────┬────────────────────────────────────┘
                                        │
                                        ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │  EXPLAINABLE AI ENGINE: SHAP GAME-THEORETIC ATTRIBUTION                │
    │  • Marginal Feature Contributions Across Sequence Window T             │
    │  • Domain Root Cause Explanation for Predicted Attack (e.g. DoS)       │
    └────────────────────────────────────────────────────────────────────────┘
    ```
    """)

    st.markdown("---")

    # 2. Mathematical Formulations
    st.markdown("### 📐 Mathematical Foundations")
    mcol1, mcol2 = st.columns(2)
    with mcol1:
        st.markdown("""
        **1. Feature Standardization (Z-Score):**
        $$X_{scaled} = \\frac{X - \\mu}{\\sigma}$$
        *Fitted strictly on training partition; applied to testing partition and inference stream.*

        **2. Balanced Class Weighting:**
        $$w_c = \\frac{N}{K \\times N_c}$$
        *Compensates for severe imbalance in minority categories (Worms, Backdoor, Shellcode).*
        """)

    with mcol2:
        st.markdown("""
        **3. Attention Weights Normalization:**
        $$\\alpha_t = \\frac{\\exp(e_t)}{\\sum_{j=1}^{T} \\exp(e_j)}, \\quad C = \\sum_{t=1}^{T} \\alpha_t h_t$$
        *Weights dynamic importance across packet observation sequence.*

        **4. SHAP Additive Attribution:**
        $$f(x) = \\phi_0 + \\sum_{i=1}^{M} \\phi_i$$
        *Quantifies positive and negative marginal feature influence on attack prediction.*
        """)

    st.markdown("---")

    # 3. Software Environment & Dependencies Manifest
    st.markdown("### 💻 Runtime Environment & Software Versions")
    env_data = [
        {"Component": "Operating System", "Version / Details": f"{platform.system()} {platform.release()} ({platform.machine()})"},
        {"Component": "Python Runtime", "Version / Details": sys.version.split()[0]},
        {"Component": "Deep Learning Engine", "Version / Details": f"Keras {keras.__version__} (Backend: {keras.backend.backend()})"},
        {"Component": "Tensor / Autograd Engine", "Version / Details": f"PyTorch {torch.__version__}"},
        {"Component": "Explainable AI (XAI)", "Version / Details": f"SHAP {shap.__version__}"},
        {"Component": "Machine Learning & Metrics", "Version / Details": f"Scikit-Learn {sklearn.__version__}"},
        {"Component": "Data Processing", "Version / Details": f"Pandas {pd.__version__}"},
        {"Component": "Interactive SOC Frontend", "Version / Details": f"Streamlit {st.__version__}"}
    ]
    st.dataframe(pd.DataFrame(env_data), use_container_width=True, hide_index=True)

    st.markdown("---")

    # 4. Defensive Charter & Research Ethics
    st.markdown("### 🛡️ Defensive IDS Research Charter")
    st.markdown("""
    * **Defensive Purpose:** This application is strictly an intrusion detection, threat telemetry, and explainable AI diagnostic system. It does not implement attack generation, port scanning, exploitation, or offensive payloads.
    * **No Fabricated Statistics:** All performance figures, confusion matrices, and SHAP attributions are computed from actual model execution on the official UNSW-NB15 benchmark dataset.
    * **Zero Data Leakage:** All preprocessing scalers and categorical encoders are fitted strictly on the training partition without access to testing records.
    """)

