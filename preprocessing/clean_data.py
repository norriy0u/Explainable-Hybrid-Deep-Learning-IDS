import sys
from pathlib import Path
_base_dir = Path(__file__).resolve().parent
while _base_dir.name != 'Explainable_Hybrid_IDS' and _base_dir.parent != _base_dir:
    _base_dir = _base_dir.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

"""
Data Cleaning and Normalization Module for Network Traffic Data.
Handles whitespace, missing values, infinite values, and category standardization.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from utils.logger import get_logger

logger = get_logger("CleanData")

# Mapping to standardize known typographical variations in UNSW-NB15 attack categories
CATEGORY_SYNONYMS = {
    "backdoors": "Backdoor",
    "backdoor": "Backdoor",
    "dos": "DoS",
    "exploits": "Exploits",
    "fuzzers": "Fuzzers",
    "generic": "Generic",
    "reconnaissance": "Reconnaissance",
    "shellcode": "Shellcode",
    "worms": "Worms",
    "analysis": "Analysis",
    "normal": "Normal"
}

def standardize_attack_categories(series: pd.Series, binary_label_series: Optional[pd.Series] = None) -> pd.Series:
    """Normalizes attack category names to exact standard naming."""
    clean_series = series.astype(str).str.strip()
    
    # Map lowercase to standard canonical names
    clean_series = clean_series.apply(
        lambda x: CATEGORY_SYNONYMS.get(x.lower(), x.title() if x.lower() not in ["nan", "none", ""] else "Normal")
    )
    
    if binary_label_series is not None:
        # If binary label indicates normal (0), ensure attack_cat is 'Normal'
        mask_normal = (binary_label_series == 0) | (clean_series.isin(["Nan", "None", "", "Normal"]))
        clean_series = clean_series.where(~mask_normal, "Normal")
        
    return clean_series

def clean_network_dataframe(
    df: pd.DataFrame,
    numerical_impute_values: Optional[Dict[str, float]] = None,
    is_training: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, float]]:
    """
    Cleans network traffic dataframe.
    Calculates median imputation values if is_training=True; otherwise applies provided impute values.
    """
    df_clean = df.copy()
    cleaning_report = {
        "initial_rows": len(df_clean),
        "inf_replaced": 0,
        "missing_imputed": 0,
        "categories_standardized": False
    }
    
    # 1. Clean String Columns (Strip Whitespace)
    str_cols = df_clean.select_dtypes(include=["object"]).columns
    for col in str_cols:
        df_clean[col] = df_clean[col].astype(str).str.strip()
        # Replace empty strings or literal 'nan' strings with NaN
        df_clean[col] = df_clean[col].replace({"": np.nan, "nan": np.nan, "None": np.nan, "-": "none" if col == "service" else "-"})
    
    # 2. Standardize Attack Category if present
    if "attack_cat" in df_clean.columns:
        binary_labels = df_clean["label"] if "label" in df_clean.columns else None
        df_clean["attack_cat"] = standardize_attack_categories(df_clean["attack_cat"], binary_labels)
        cleaning_report["categories_standardized"] = True

    # 3. Numeric Columns: Replace Infs with NaN
    num_cols = df_clean.select_dtypes(include=["number"]).columns
    inf_count = 0
    for col in num_cols:
        infinities = np.isinf(df_clean[col])
        if infinities.any():
            inf_count += int(infinities.sum())
            df_clean.loc[infinities, col] = np.nan
    cleaning_report["inf_replaced"] = inf_count

    # 4. Impute Missing Values in Numeric Columns using Median
    computed_medians = numerical_impute_values or {}
    if is_training:
        computed_medians = {}
        for col in num_cols:
            median_val = float(df_clean[col].median(skipna=True))
            if np.isnan(median_val):
                median_val = 0.0
            computed_medians[col] = median_val
            
    # Apply imputation
    for col, med_val in computed_medians.items():
        if col in df_clean.columns and df_clean[col].isnull().any():
            missing_c = int(df_clean[col].isnull().sum())
            cleaning_report["missing_imputed"] += missing_c
            df_clean[col] = df_clean[col].fillna(med_val)

    # Fill any remaining categorical NaNs with 'unknown'
    for col in str_cols:
        if col in df_clean.columns and df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna("unknown")

    logger.info(
        f"Cleaned dataset: {len(df_clean)} rows. "
        f"Infs replaced: {cleaning_report['inf_replaced']}, "
        f"Missings imputed: {cleaning_report['missing_imputed']}."
    )
    return df_clean, cleaning_report, computed_medians
