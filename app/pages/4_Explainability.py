"""Explainability — global and local model explanations.

Phase 3: fully wired to Sprint 5's pre-generated explainability
artifacts through `services.explainability_service`. No SHAP value is
recomputed on this page — every figure and factor shown here was
generated once, offline, and is only read/displayed here.
"""

import logging
import sys
import time
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parents[1]
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import pandas as pd
import streamlit as st

from components.cards import info_box, placeholder_card
from components.charts import chart_placeholder
from components.layout import footer, page_title, section_header
from components.tables import data_table
from services.explainability_service import get_feature_importance, get_shap_summary
from state import get_app_state
from utils.paths import get_scenario_waterfall_path

logger = logging.getLogger(__name__)

_page_start = time.perf_counter()
logger.info("Loading Explainability page...")

page_title(
    "Explainability",
    subtitle="How the system explains its decisions — globally across all transactions, and locally for a single one.",
)

info_box(
    "SHAP (SHapley Additive exPlanations) estimates how much each feature contributed to a "
    "prediction: a <strong>positive contribution</strong> pushes the prediction toward Fraud, a "
    "<strong>negative contribution</strong> pushes it toward Not Fraud. Contributions describe how "
    "the model reasoned — they are not proof that a feature caused fraud. Every figure below was "
    "generated once by the system's offline SHAP analysis; nothing on this page recomputes it.",
    kind="info",
)

# --- Global Feature Importance -------------------------------------------------
section_header(
    "Feature Importance",
    "Global view — the features the trained model relies on most, measured across all transactions "
    "it was trained on.",
)
importance_rows = get_feature_importance()
logger.info("Loaded feature importance: %d rows.", len(importance_rows))

if not importance_rows:
    info_box("Feature importance is not available yet.", kind="warning")
else:
    top_importance = sorted(importance_rows, key=lambda r: r.importance, reverse=True)[:15]
    col1, col2 = st.columns([3, 2])
    with col1:
        chart_df = pd.DataFrame([{"feature": r.feature, "importance": r.importance} for r in top_importance])
        st.bar_chart(chart_df.set_index("feature"), horizontal=True)
    with col2:
        data_table([{"Feature": r.feature, "Importance": round(r.importance, 4)} for r in top_importance])

# --- SHAP Summary ----------------------------------------------------------
section_header(
    "SHAP Summary",
    "Global view — each dot is one transaction; its position shows how strongly that feature "
    "pushed the prediction toward Fraud (right) or Not Fraud (left).",
)
shap_summary = get_shap_summary()
logger.info(
    "Loaded SHAP summary: %d top features, error=%s.", len(shap_summary.top_features), shap_summary.error
)

if shap_summary.error:
    info_box(f"SHAP summary is not available: {shap_summary.error}", kind="warning")
else:
    col1, col2 = st.columns(2)
    with col1:
        if shap_summary.summary_plot_path and Path(shap_summary.summary_plot_path).exists():
            st.image(shap_summary.summary_plot_path, caption="SHAP Summary Plot", use_container_width=True)
        else:
            chart_placeholder("SHAP summary plot unavailable", size="md")
    with col2:
        if shap_summary.beeswarm_plot_path and Path(shap_summary.beeswarm_plot_path).exists():
            st.image(shap_summary.beeswarm_plot_path, caption="SHAP Beeswarm Plot", use_container_width=True)
        else:
            chart_placeholder("SHAP beeswarm plot unavailable", size="md")

    if shap_summary.bar_plot_path and Path(shap_summary.bar_plot_path).exists():
        st.image(shap_summary.bar_plot_path, caption="Mean |SHAP Value| by Feature", use_container_width=True)

    if shap_summary.top_features:
        with st.container(border=True):
            st.markdown("**Top Features by Mean |SHAP| Value**")
            for row in shap_summary.top_features[:5]:
                st.markdown(f"- `{row.feature}` — {row.importance:.4f}")

# --- SHAP Dependence Plots ---------------------------------------------------
section_header(
    "SHAP Dependence Plots",
    "Global view — how a single feature's contribution to the prediction changes as its value varies.",
)
if not shap_summary.error and shap_summary.dependence_plot_paths:
    cols = st.columns(len(shap_summary.dependence_plot_paths))
    for col, (feature, path) in zip(cols, shap_summary.dependence_plot_paths.items()):
        with col:
            if Path(path).exists():
                st.image(path, caption=f"Dependence: {feature}", use_container_width=True)
            else:
                chart_placeholder(f"Dependence plot for {feature} unavailable", size="sm")
else:
    chart_placeholder("Dependence plots unavailable", size="md")

# --- Local Explanation Example -----------------------------------------------
section_header(
    "Local Explanation Example",
    "Local view — a complete explanation of one representative fraud case from the business "
    "scenario validation. The waterfall shows how each feature pushed this single prediction from "
    "the model's baseline to its final decision — the same explanation an analyst sees on the "
    "Single Transaction page.",
)

app_state = get_app_state()
example_scenario = next(
    (s for s in app_state.validation_summary.scenarios if s.get("scenario_id") == "F1"), None
)
logger.info("Local explanation example loaded: scenario_found=%s", example_scenario is not None)

if example_scenario is None:
    placeholder_card("Local Explanation Example", "No validation scenario available to illustrate.")
else:
    col1, col2 = st.columns([1, 1])
    with col1:
        waterfall_path = get_scenario_waterfall_path(example_scenario["scenario_id"])
        if waterfall_path.exists():
            st.image(str(waterfall_path), caption=f"Scenario {example_scenario['scenario_id']}: {example_scenario['title']}", use_container_width=True)
        else:
            chart_placeholder("Waterfall plot unavailable", size="md")
    with col2:
        with st.container(border=True):
            st.markdown(f"**{example_scenario['title']}**")
            st.markdown(f"Prediction: **{example_scenario['prediction']}** ({example_scenario['confidence_pct']}% confidence)")
            st.markdown("Top contributing factors:")
            for factor in str(example_scenario.get("top_shap_factors", "")).split(";"):
                factor = factor.strip()
                if factor:
                    st.markdown(f"- {factor}")
            st.markdown(f"Recommendation: {example_scenario.get('recommendation_actions', 'N/A')}")

elapsed = time.perf_counter() - _page_start
logger.info("Explainability page loaded in %.3fs.", elapsed)

footer()
