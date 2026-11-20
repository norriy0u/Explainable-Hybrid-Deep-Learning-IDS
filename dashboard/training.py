"""
Dashboard Page 3: Model Training and Hyperparameter Tuning.
Allows configuration, training execution, class weight inspection, loss curve tracking, and model reload.
"""

from pathlib import Path
import streamlit as st
import pandas as pd

from config.config import (
    MODELS_DIR,
    PREPROCESSORS_DIR,
    REPORTS_DIR,
    PLOTS_DIR,
    DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_SEQUENCE_LENGTH,
    DEFAULT_CNN_FILTERS,
    DEFAULT_LSTM_UNITS,
    DEFAULT_DROPOUT_RATE,
    DEFAULT_DENSE_UNITS,
    TRAIN_DATASET_PATH,
    TEST_DATASET_PATH,
    BENCHMARK_SAMPLE_PATH
)
from utils.helpers import load_json


def render_training_page():
    st.markdown("""
    <div class="soc-header">
        <h1 class="soc-header-title">⚡ MODEL TRAINING & ARCHITECTURAL CONFIGURATION</h1>
        <div class="soc-header-subtitle">
            Configure Hyperparameters, Train Hybrid CNN-BiLSTM-Attention IDS, and Monitor Convergence
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Active Model Status
    model_path = MODELS_DIR / "hybrid_ids_model.keras"
    pipeline_path = PREPROCESSORS_DIR / "preprocessing_pipeline.joblib"
    history_path = REPORTS_DIR / "training_history.json"
    curves_path = PLOTS_DIR / "training_curves.png"
    scol1, scol2 = st.columns([1.2, 0.8])
    with scol1:
        if model_path.exists() and pipeline_path.exists():
            st.success(f"Active Trained Model Available: `{model_path.name}` ({model_path.stat().st_size / (1024*1024):.2f} MB)")
        else:
            st.warning("No active trained model found. Configure hyperparameters below and train the model.")

    with scol2:
        if history_path.exists():
            h_data = load_json(history_path)
            epochs_done = len(h_data.get("loss", []))
            best_val_acc = max(h_data.get("val_accuracy", [0.0]))
            st.info(f"Last Training: {epochs_done} Epochs | Peak Val Accuracy: **{best_val_acc * 100:.2f}%**")

    st.markdown("---")

    # 2. Hyperparameter Configuration Grid
    st.markdown("### 🛠️ Training Hyperparameters")
    
    hcol1, hcol2, hcol3 = st.columns(3)
    with hcol1:
        epochs = st.slider("Max Epochs:", min_value=2, max_value=50, value=DEFAULT_EPOCHS, step=1)
        batch_size = st.select_slider("Batch Size:", options=[32, 64, 128, 256, 512], value=DEFAULT_BATCH_SIZE)
        learning_rate = st.select_slider("Learning Rate:", options=[0.01, 0.005, 0.001, 0.0005, 0.0001], value=DEFAULT_LEARNING_RATE)

    with hcol2:
        seq_len = st.slider("Sequence Length (T):", min_value=3, max_value=10, value=DEFAULT_SEQUENCE_LENGTH, step=1)
        cnn_filters = st.select_slider("CNN Filters:", options=[16, 32, 64, 128], value=DEFAULT_CNN_FILTERS)
        lstm_units = st.select_slider("BiLSTM Units:", options=[16, 32, 64, 128], value=DEFAULT_LSTM_UNITS)

    with hcol3:
        dropout = st.slider("Dropout Rate:", min_value=0.1, max_value=0.5, value=DEFAULT_DROPOUT_RATE, step=0.05)
        use_weights = st.checkbox("Apply Balanced Class Weighting (w_c = N / (K * N_c))", value=True)
        mode = st.radio("Execution Dataset Mode:", ["Fast Dev Mode (Subsampled)", "Full Benchmark Partition (Comprehensive)"])

    subsample_n = 20000 if "Fast" in mode else None

    if not TRAIN_DATASET_PATH.exists() or not TEST_DATASET_PATH.exists():
        st.info("ℹ️ **Cloud Environment Notice:** The raw 175k-record dataset is excluded from the cloud repository to prevent container memory exhaustion. Training will execute on the bundled 1,000-record benchmark sample.")

    # 3. Training Execution Action
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Launch Hybrid IDS Training Pipeline", use_container_width=True):
        progress_bar = st.progress(0, text="Initializing preprocessing and sequence construction...")
        status_box = st.empty()
        
        try:
            status_box.info("Loading deep learning training engine and preprocessing pipeline...")
            progress_bar.progress(15, text="Importing Keras/PyTorch training modules...")
            from training.train_model import train_hybrid_ids

            status_box.info("Fitting Preprocessing Pipeline and Generating Flow Sequences...")
            progress_bar.progress(25, text="Processing UNSW-NB15 dataset...")
            
            model, pipeline, eval_res = train_hybrid_ids(
                sequence_length=seq_len,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                filters=cnn_filters,
                lstm_units=lstm_units,
                dropout_rate=dropout,
                use_class_weights=use_weights,
                sample_subsample=subsample_n
            )
            
            progress_bar.progress(100, text="Training and Test Evaluation Complete!")
            status_box.success(
                f"Model training complete in {eval_res['training_time_seconds']}s! "
                f"Test Accuracy: {eval_res['accuracy']*100:.2f}%, "
                f"Macro F1: {eval_res['macro_f1']*100:.2f}%, "
                f"DoS F1: {eval_res['dos_performance'].get('f1_score', 0)*100:.2f}%"
            )
            st.balloons()
        except Exception as e:
            status_box.error(f"Training encountered an error: {str(e)}")

    # 4. Display Existing Training Curves & Artifacts
    if curves_path.exists():
        st.markdown("---")
        st.markdown("### 📈 Training & Validation Convergence Curves")
        st.image(str(curves_path), width="stretch", caption="Model Loss and Accuracy Trajectory Across Epochs")

