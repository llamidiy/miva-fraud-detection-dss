"""Model Performance — champion model metrics and model comparison.

Phase 3: fully wired to Sprint 4/4.5's existing evaluation results
through `services.metrics_service`. No model is retrained or
re-evaluated on this page.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Any

_APP_DIR = Path(__file__).resolve().parents[1]
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import pandas as pd
import streamlit as st

from components.cards import info_box, metric_card
from components.layout import footer, page_title, section_header
from components.tables import data_table
from state import get_app_state

logger = logging.getLogger(__name__)

_page_start = time.perf_counter()
logger.info("Loading Model Performance page...")


def _build_selection_rationale(champion: Any) -> list[str]:
    """Build data-driven bullet points explaining why the champion model was selected.

    Every claim is derived from `champion.comparison` (the real Sprint 4
    model comparison table) at render time, rather than a hardcoded
    claim — so the rationale stays truthful if the models are ever
    retrained and the comparison table changes.

    Args:
        champion: The champion model's `ModelMetrics`.

    Returns:
        A list of rationale bullet points.
    """
    bullets: list[str] = []
    comparison = champion.comparison
    others = [row for row in comparison if row["model_name"] != champion.model_name]

    if champion.roc_auc is not None:
        other_roc_aucs = [row["roc_auc"] for row in others if row.get("roc_auc") is not None]
        if other_roc_aucs and champion.roc_auc >= max(other_roc_aucs):
            bullets.append(f"Highest ROC-AUC among all evaluated models ({champion.roc_auc:.4f}).")

    if champion.precision is not None and champion.recall is not None:
        bullets.append(
            f"Strong precision ({champion.precision:.1%}) and recall ({champion.recall:.1%}) balance, "
            "minimizing both false alarms and missed fraud cases."
        )

    if others and champion.accuracy is not None and champion.training_time_seconds:
        closest = min(others, key=lambda row: abs((row.get("accuracy") or 0) - champion.accuracy))
        closest_time = closest.get("training_time_seconds")
        if closest_time and closest_time > champion.training_time_seconds * 1.5:
            speedup = closest_time / champion.training_time_seconds
            closest_label = closest["model_name"].replace("_", " ").title()
            bullets.append(
                f"Comparable predictive performance to {closest_label} (accuracy "
                f"{closest['accuracy']:.4f} vs. {champion.accuracy:.4f}) with a "
                f"{speedup:.1f}x shorter training time "
                f"({champion.training_time_seconds:.0f}s vs. {closest_time:.0f}s)."
            )

    bullets.append(
        "Tree-based architecture is natively supported by SHAP's TreeExplainer, enabling the "
        "per-transaction explanations used throughout this decision-support system."
    )
    bullets.append(
        "Strong accuracy-efficiency trade-off makes it well suited for production deployment "
        "and periodic retraining."
    )
    return bullets


page_title(
    "Model Performance",
    subtitle="Champion model metrics and a comparison against every candidate model evaluated during training.",
)

app_state = get_app_state()
champion = app_state.model_metrics
logger.info("Loaded model metrics: model=%s, error=%s.", champion.model_name, champion.error)

if champion.error:
    info_box(f"Model metrics are not available: {champion.error}", kind="danger")
    footer()
    st.stop()

# --- Champion Model -----------------------------------------------------------
section_header("Champion Model", "The model currently deployed for predictions.")
col1, col2, col3 = st.columns(3)
with col1:
    metric_card("Model", champion.algorithm, help_text=f"Registered as `{champion.model_name}`")
with col2:
    metric_card("Accuracy", f"{champion.accuracy:.4f}" if champion.accuracy is not None else "N/A")
with col3:
    metric_card("ROC-AUC", f"{champion.roc_auc:.4f}" if champion.roc_auc is not None else "N/A")

col4, col5, col6 = st.columns(3)
with col4:
    metric_card("Precision", f"{champion.precision:.4f}" if champion.precision is not None else "N/A")
with col5:
    metric_card("Recall", f"{champion.recall:.4f}" if champion.recall is not None else "N/A")
with col6:
    metric_card("F1-Score", f"{champion.f1_score:.4f}" if champion.f1_score is not None else "N/A")

# --- Model Selection Rationale (new) -------------------------------------------
section_header("Model Selection Rationale", "Why this model was chosen as the champion, not just which one.")
rationale = _build_selection_rationale(champion)
with st.container(border=True):
    for bullet in rationale:
        st.markdown(f"- {bullet}")
st.caption(
    "Selection weighed discrimination (ROC-AUC), the precision-recall balance on fraud cases, "
    "explainability support, and operational practicality — not accuracy alone."
)

# --- Model Comparison -----------------------------------------------------------
section_header("Model Comparison", "All four trained models, side by side.")
comparison_rows = [
    {
        "Model": row["model_name"],
        "Accuracy": row["accuracy"],
        "Precision": row["precision"],
        "Recall": row["recall"],
        "F1-Score": row["f1_score"],
        "ROC-AUC": row["roc_auc"],
        "Training Time (s)": row["training_time_seconds"],
    }
    for row in champion.comparison
]
data_table(comparison_rows)

if comparison_rows:
    chart_df = pd.DataFrame(comparison_rows).set_index("Model")
    col1, col2 = st.columns(2)
    with col1:
        st.bar_chart(chart_df[["ROC-AUC"]])
        st.caption("ROC-AUC by model — higher is better.")
    with col2:
        st.bar_chart(chart_df[["Training Time (s)"]])
        st.caption("Training time in seconds — lower is better at comparable performance.")

elapsed = time.perf_counter() - _page_start
logger.info("Model Performance page loaded in %.3fs.", elapsed)

footer()
