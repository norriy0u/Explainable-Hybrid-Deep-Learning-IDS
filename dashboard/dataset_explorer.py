"""
Dashboard Page 2: UNSW-NB15 Dataset Explorer.
Allows inspection, upload, exploratory data analysis, class distribution, and schema validation.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

from config.config import (
    DATASET_DIR,
    TRAIN_DATASET_PATH,
    TEST_DATASET_PATH,
    BENCHMARK_SAMPLE_PATH,
    FEATURES_METADATA_PATH,
    LABEL_MAPPING,
    DEMO_ATTACK_CLASS
)
from preprocessing.load_dataset import load_csv_data, validate_schema

def render_dataset_explorer_page():
    st.markdown("""
    <div class="soc-header">
        <h1 class="soc-header-title">📁 DATASET EXPLORER — UNSW-NB15 REPOSITORY</h1>
        <div class="soc-header-subtitle">
            Multiclass Network Traffic Corpus Inspection, Schema Auditing, and Imbalance Profiling
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Dataset Selection / Upload
    st.markdown("### 🗃️ Select or Upload Dataset Split")
    source_choice = st.radio(
        "Choose Dataset Source:",
        ["Official Benchmark Repository (Local)", "Upload Custom Network Capture CSV"],
        horizontal=True
    )

    df_selected = None
    split_name = ""

    if source_choice == "Official Benchmark Repository (Local)":
        split_options = [
            f"Pre-Packaged Benchmark Sample ({BENCHMARK_SAMPLE_PATH.name} — 1,000 records)",
            f"Testing Partition ({TEST_DATASET_PATH.name})",
            f"Training Partition ({TRAIN_DATASET_PATH.name})"
        ]
        split_option = st.selectbox("Select Benchmark Partition:", split_options)

        if "sample" in split_option.lower():
            target_path = BENCHMARK_SAMPLE_PATH
        elif "training" in split_option.lower():
            target_path = TRAIN_DATASET_PATH
        else:
            target_path = TEST_DATASET_PATH

        train_sample_path = DATASET_DIR / "train_benchmark_sample.csv"
        test_sample_path = DATASET_DIR / "test_benchmark_sample.csv"

        if target_path.exists():
            df_selected = load_csv_data(target_path)
            split_name = target_path.name
        elif "training" in split_option.lower() and train_sample_path.exists():
            st.info("ℹ️ **Cloud Environment Notice:** Loaded official UNSW-NB15 Training Partition (Stratified Benchmark Sample — 2,879 records) displaying genuine class imbalance.")
            df_selected = load_csv_data(train_sample_path)
            split_name = "UNSW-NB15 Training Partition (Stratified Sample)"
        elif "testing" in split_option.lower() and test_sample_path.exists():
            st.info("ℹ️ **Cloud Environment Notice:** Loaded official UNSW-NB15 Testing Partition (Stratified Benchmark Sample — 3,503 records) displaying genuine class imbalance.")
            df_selected = load_csv_data(test_sample_path)
            split_name = "UNSW-NB15 Testing Partition (Stratified Sample)"
        elif BENCHMARK_SAMPLE_PATH.exists():
            st.info(f"Loaded balanced benchmark sample ({BENCHMARK_SAMPLE_PATH.name} — 1,000 records, 100 per category).")
            df_selected = load_csv_data(BENCHMARK_SAMPLE_PATH)
            split_name = f"Benchmark Sample ({BENCHMARK_SAMPLE_PATH.name})"
        else:
            st.error(f"File not found: {target_path}. Please place official CSVs in the dataset/ folder.")
    else:
        uploaded_file = st.file_uploader("Upload Network Traffic CSV:", type=["csv"])
        if uploaded_file is not None:
            df_selected = pd.read_csv(uploaded_file, low_memory=False)
            split_name = uploaded_file.name
        else:
            st.info("Upload a CSV file to inspect its feature schema and class distributions.")

    if df_selected is not None:
        schema_info = validate_schema(df_selected)

        # 2. Key Dataset Metrics
        st.markdown("---")
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("Total Records", f"{schema_info['num_records']:,}")
        mcol2.metric("Total Features", schema_info["num_features"])
        mcol3.metric("Numeric Features", len(schema_info["numeric_columns"]))
        mcol4.metric("Categorical Features", len(schema_info["categorical_columns"]))

        # 3. Class Distribution Profiling
        st.markdown("### 🎯 Class Distribution Profiling (10-Class Target)")
        if "attack_cat" in df_selected.columns:
            dist_counts = df_selected["attack_cat"].astype(str).str.strip().value_counts().reset_index()
            dist_counts.columns = ["Attack Category", "Record Count"]
            dist_counts["Percentage"] = (dist_counts["Record Count"] / len(df_selected)) * 100

            dcol_chart, dcol_table = st.columns([1.3, 0.7])
            with dcol_chart:
                fig_bar = px.bar(
                    dist_counts,
                    x="Attack Category",
                    y="Record Count",
                    color="Attack Category",
                    text_auto=True,
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
                fig_bar.update_layout(
                    title=f"<b>Attack Distribution in {split_name}</b>",
                    paper_bgcolor="#111827",
                    plot_bgcolor="#1F2937",
                    font=dict(color="#E5E7EB"),
                    xaxis_title="Category",
                    yaxis_title="Count",
                    showlegend=False
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            with dcol_table:
                st.markdown("**Distribution Data Table**")
                display_df = dist_counts.copy()
                display_df["Percentage"] = display_df["Percentage"].map(lambda x: f"{x:.2f}%")
                st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.warning("Selected dataset does not contain an 'attack_cat' target column.")

        # 4. Data Preview & Feature Inspector
        st.markdown("---")
        st.markdown("### 🔍 Raw Feature Inspection")
        max_rows = st.slider("Preview Rows:", min_value=5, max_value=100, value=15, step=5)
        st.dataframe(df_selected.head(max_rows), use_container_width=True)

        # 5. Missing & Duplicate Analysis
        st.markdown("### 🛡️ Data Quality & Integrity Diagnostics")
        qcol1, qcol2 = st.columns(2)
        with qcol1:
            st.markdown(f"**Missing Values Total:** `{schema_info['total_missing_values']}`")
            if schema_info["missing_per_column"]:
                st.json(schema_info["missing_per_column"])
            else:
                st.success("No missing (NaN) values detected across dataset columns.")

        with qcol2:
            st.markdown(f"**Duplicate Records:** `{schema_info['duplicate_rows']:,}`")
            if schema_info["duplicate_rows"] > 0:
                st.info("Duplicate network observations are standard in flow-aggregated captures.")
            else:
                st.success("All records represent unique connection flows.")

