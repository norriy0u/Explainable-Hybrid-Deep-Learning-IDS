import sys
from pathlib import Path
_base_dir = Path(__file__).resolve().parent
while 'config' not in [p.name for p in _base_dir.iterdir() if p.is_dir()] and _base_dir.parent != _base_dir:
    _base_dir = _base_dir.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

"""
SHAP Feature Contribution and Explanation Visualization Module.
Generates academic publication-quality matplotlib plots and interactive Plotly charts.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

from utils.logger import get_logger
from config.config import SHAP_PLOTS_DIR

logger = get_logger("FeatureVisualization")

def plot_shap_bar_chart(
    explanation_result: Dict[str, Any],
    save_path: Optional[Path] = None
) -> Path:
    """
    Renders and saves a matplotlib horizontal bar chart showing top positive and negative SHAP contributions.
    """
    save_path = save_path or (SHAP_PLOTS_DIR / "shap_feature_importance.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    features_info = explanation_result["top_features"][::-1] # reverse so highest is on top
    features = [f["feature"] for f in features_info]
    shap_vals = [f["shap_value"] for f in features_info]
    colors = ["#EF4444" if v >= 0 else "#10B981" for v in shap_vals]
    
    plt.figure(figsize=(10, 6), dpi=150)
    plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
    
    bars = plt.barh(features, shap_vals, color=colors, height=0.65)
    plt.axvline(0, color="#6B7280", linestyle="--", linewidth=1.2)
    
    explained_class = explanation_result["explained_class_name"]
    plt.title(f"SHAP Feature Attributions for {explained_class} Prediction", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("SHAP Value (Net Impact on Model Output)", fontsize=11, labelpad=8)
    plt.ylabel("Network Traffic Feature", fontsize=11)
    
    # Annotate values
    for bar in bars:
        width = bar.get_width()
        ha = "left" if width >= 0 else "right"
        offset = 0.001 if width >= 0 else -0.001
        plt.text(width + offset, bar.get_y() + bar.get_height() / 2,
                 f"{width:+.4f}",
                 va="center", ha=ha, fontsize=9, fontweight="medium",
                 color="#111827")
                 
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved SHAP bar plot to: {save_path}")
    return save_path

def create_interactive_shap_bar_chart(explanation_result: Dict[str, Any]) -> go.Figure:
    """
    Creates an interactive Plotly horizontal bar chart for the Streamlit dashboard.
    """
    raw_info = explanation_result.get("top_features", [])
    features_info = [f for f in raw_info if isinstance(f, dict)][::-1]
    
    features = [f.get("feature", "Unknown") for f in features_info]
    shap_vals = [float(f.get("shap_value", 0.0)) for f in features_info]
    feature_vals = [float(f.get("feature_value", f.get("observed_value", 0.0))) for f in features_info]
    directions = [f.get("direction", "Increased Risk" if v >= 0 else "Decreased Risk") for f, v in zip(features_info, shap_vals)]
    contexts = [f.get("cyber_context", "") for f in features_info]

    colors = ["#EF4444" if v >= 0 else "#10B981" for v in shap_vals]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=features,
        x=shap_vals,
        orientation="h",
        marker=dict(color=colors, line=dict(color="#1F2937", width=1)),
        customdata=list(zip(feature_vals, directions, contexts)),
        hovertemplate=(
            "<b>%{y}</b><br>" +
            "SHAP Attribution: %{x:+.5f}<br>" +
            "Observed Value: %{customdata[0]:.4f}<br>" +
            "Impact: %{customdata[1]}<br>" +
            "<i>%{customdata[2]}</i><extra></extra>"
        )
    ))

    fig.add_vline(x=0, line_width=1.5, line_dash="dash", line_color="#9CA3AF")

    explained_name = explanation_result.get("explained_class_name", "Attack")
    fig.update_layout(
        title=dict(
            text=f"<b>SHAP Feature Influence for {explained_name} Classification</b>",
            font=dict(size=15, color="#F3F4F6")
        ),
        paper_bgcolor="#111827",
        plot_bgcolor="#1F2937",
        font=dict(color="#E5E7EB", family="Inter, system-ui, sans-serif"),
        xaxis=dict(
            title="SHAP Marginal Contribution",
            gridcolor="#374151",
            zerolinecolor="#4B5563"
        ),
        yaxis=dict(
            title="Feature",
            gridcolor="#374151"
        ),
        margin=dict(l=80, r=40, t=50, b=40),
        height=450
    )
    return fig

def create_temporal_attention_chart(
    attention_weights: np.ndarray,
    sequence_length: int = 5
) -> go.Figure:
    """
    Creates an interactive line/area chart showing the attention distribution across sequence time steps.
    """
    time_steps = [f"Step {t+1} (t-{sequence_length - t - 1})" for t in range(sequence_length)]
    weights = [float(w) for w in attention_weights]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_steps,
        y=weights,
        mode="lines+markers+text",
        text=[f"{w*100:.1f}%" for w in weights],
        textposition="top center",
        line=dict(color="#06B6D4", width=3),
        marker=dict(size=9, color="#38BDF8"),
        fill="tozeroy",
        fillcolor="rgba(6, 182, 212, 0.2)"
    ))

    fig.update_layout(
        title=dict(
            text="<b>Temporal Attention Weights Across Window (α_t)</b>",
            font=dict(size=13, color="#F3F4F6")
        ),
        paper_bgcolor="#111827",
        plot_bgcolor="#1F2937",
        font=dict(color="#E5E7EB"),
        xaxis=dict(gridcolor="#374151", title="Time Step in Sequence"),
        yaxis=dict(gridcolor="#374151", title="Attention Weight (α_t)", range=[0, max(max(weights) * 1.32, 0.1)]),
        margin=dict(l=50, r=25, t=48, b=40),
        height=350
    )
    return fig
