"""Dashboard — system overview and headline metrics.

The entry point of the application. Within a few seconds a new user
should understand what the system is, what model powers it, how it was
validated, and which workflow to open next. All metrics are live,
loaded through the service layer (`state.get_app_state`).
"""

import logging
import sys
import time
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parents[1]
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import streamlit as st

from components.cards import info_box, metric_card, status_badge_html
from components.layout import footer, page_title, section_header
from state import get_app_state

logger = logging.getLogger(__name__)

_PAGES_DIR = _APP_DIR / "pages"

_page_start = time.perf_counter()
logger.info("Loading Dashboard page...")

page_title(
    "Dashboard",
    subtitle="Machine learning-based fraud detection and decision support for digital banking transactions.",
)

st.markdown(
    "This system screens banking transactions with a validated machine learning model, "
    "recommends an analyst action for every prediction, and explains each decision using "
    "SHAP-based model interpretation."
)

app_state = get_app_state()

if app_state.system_status != "Operational":
    info_box(
        "Some backend data could not be loaded — see individual sections below for details.",
        kind="warning",
    )

# --- System Overview -----------------------------------------------------------
section_header("System Overview", "Headline metrics for the deployed champion model.")

summary = app_state.validation_summary
matched_help = (
    f"{summary.passed}/{summary.total_scenarios} executed successfully · "
    f"{summary.matches} matched the expected outcome"
    if summary.total_scenarios
    else None
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    metric_card(
        "Champion Model",
        "XGBoost",
        help_text=f"{app_state.model_metrics.algorithm}, registered as `{app_state.model_metrics.model_name}`",
    )
with col2:
    roc_auc = app_state.model_metrics.roc_auc
    metric_card(
        "ROC-AUC",
        f"{roc_auc:.4f}" if roc_auc is not None else "Unavailable",
        help_text="Discrimination between fraud and legitimate transactions on held-out test data",
    )
with col3:
    metric_card(
        "Validation Scenarios",
        str(summary.total_scenarios),
        help_text=matched_help,
    )
with col4:
    status_kind = "success" if app_state.system_status == "Operational" else "danger"
    metric_card("System Status", status_badge_html(app_state.system_status, kind=status_kind))

# --- Core Capabilities ----------------------------------------------------------
section_header("Core Capabilities", "What this system helps a fraud analyst do.")
col1, col2, col3 = st.columns(3)
with col1:
    with st.container(border=True):
        st.markdown("**Predict**")
        st.markdown("Is this transaction fraudulent? The champion model scores each transaction with a confidence level.")
with col2:
    with st.container(border=True):
        st.markdown("**Decide**")
        st.markdown("What should the analyst do? Every prediction comes with a recommended action, tiered by confidence.")
with col3:
    with st.container(border=True):
        st.markdown("**Explain**")
        st.markdown("Why did the model decide this? SHAP analysis shows which features contributed to each prediction.")

# --- Start a Workflow -----------------------------------------------------------
section_header("Start a Workflow", "Jump straight into an assessment or explore how the system reasons.")
col1, col2, col3 = st.columns(3)
with col1:
    st.page_link(str(_PAGES_DIR / "2_Single_Transaction.py"), label="Assess a Single Transaction", icon="🔍")
    st.caption("Enter one transaction and get a prediction, recommendation, and full explanation.")
with col2:
    st.page_link(str(_PAGES_DIR / "3_Batch_Assessment.py"), label="Assess a Batch of Transactions", icon="📂")
    st.caption("Upload a CSV, review it, and download a traceable assessment record.")
with col3:
    st.page_link(str(_PAGES_DIR / "4_Explainability.py"), label="Explore Model Explainability", icon="🧠")
    st.caption("See which features drive the model's decisions, globally and per transaction.")

elapsed = time.perf_counter() - _page_start
logger.info("Dashboard page loaded in %.3fs.", elapsed)

footer()
