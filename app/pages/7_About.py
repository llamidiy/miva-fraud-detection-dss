"""About — project background, methodology, architecture, and system status.

Phase 3: Limitations and Future Work are reused directly from Sprint
5.5.1's executive summary rather than rewritten, so this page stays
consistent with the dissertation's own findings.
"""

import logging
import sys
import time
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parents[1]
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import streamlit as st

from components.cards import info_box
from components.layout import footer, page_title, section_header
from state import get_app_state
from utils.markdown_utils import extract_markdown_section
from utils.paths import get_diagram_path

logger = logging.getLogger(__name__)

_page_start = time.perf_counter()
logger.info("Loading About page...")

page_title("About This System", subtitle="Background, methodology, and architecture.")

# --- Research Title / Programme / University -----------------------------------
section_header("Research Title")
st.markdown(
    "**A Machine Learning-Based Fraud Detection Decision Support System: "
    "Design, Explainability, and Business Scenario Validation**"
)

section_header("Programme & University")
st.markdown("Master of Information Technology — MIVA Open University")

# --- Objectives ------------------------------------------------------------
section_header("Objectives")
st.markdown(
    """
1. Build an accurate, reproducible fraud detection model from raw transaction data.
2. Engineer features that capture known fraud signatures (balance inconsistencies, ledger errors).
3. Compare multiple modeling approaches under a shared, fair evaluation workflow.
4. Make the champion model's decisions explainable to a non-technical reviewer, not just accurate.
5. Validate the complete system against realistic, data-grounded business scenarios.
6. Present the system as an interactive decision support tool for fraud analysts.
    """
)

# --- Methodology -------------------------------------------------------------
section_header("Methodology")
st.markdown(
    """
1. **Data Exploration** — exploratory analysis of the PaySim synthetic transaction dataset.
2. **Preprocessing & Feature Engineering** — cleaning, validation, and engineered
   features (balance deltas, ledger-error signals, log-scaled amount, time features).
3. **Model Training** — Logistic Regression, Random Forest, XGBoost, and Isolation
   Forest trained and compared under a shared, reproducible workflow with SMOTE-based
   class balancing.
4. **Explainability** — SHAP-based global and local explanations for the champion model.
5. **Validation** — the trained system exercised against ~20 realistic, data-grounded
   business scenarios covering fraud, legitimate, and edge-case transactions.
6. **Presentation Layer** — this Streamlit application, connected to every stage above
   through a dedicated service layer.
    """
)

# --- Technology Stack ---------------------------------------------------------
section_header("Technology Stack")
st.markdown(
    """
- **Language & data:** Python, pandas, NumPy
- **Modeling:** scikit-learn, XGBoost, imbalanced-learn (SMOTE)
- **Explainability:** SHAP
- **Interface:** Streamlit
    """
)

# --- Architecture Diagrams -----------------------------------------------------
section_header("Architecture", "How data flows through the system, and how the application itself is layered.")
system_diagram = get_diagram_path("system_architecture.png")
app_diagram = get_diagram_path("app_architecture.png")

if system_diagram.exists():
    st.image(str(system_diagram), caption="End-to-end pipeline", use_container_width=True)
else:
    info_box("System architecture diagram not found.", kind="warning")

if app_diagram.exists():
    st.image(str(app_diagram), caption="Streamlit application layers", use_container_width=True)
else:
    info_box("Application architecture diagram not found.", kind="warning")

# --- Limitations & Future Work (reused from the executive summary) -------------
app_state = get_app_state()
executive_summary = app_state.executive_summary
logger.info("Loading Limitations/Future Work from executive summary: available=%s.", executive_summary.available)

section_header("Limitations", "Drawn directly from the system's validation executive summary.")
if executive_summary.available:
    limitations = extract_markdown_section(executive_summary.content, "7. Limitations")
    if limitations:
        st.markdown(limitations)
    else:
        info_box("Limitations section not found in the executive summary.", kind="warning")
else:
    info_box(f"Executive summary is not available: {executive_summary.error}", kind="warning")

section_header("Future Work", "Drawn directly from the validation executive summary's recommendations.")
if executive_summary.available:
    recommendations = extract_markdown_section(executive_summary.content, "8. Recommendations")
    if recommendations:
        st.markdown(recommendations)
    else:
        info_box("Recommendations section not found in the executive summary.", kind="warning")
else:
    info_box(f"Executive summary is not available: {executive_summary.error}", kind="warning")

# --- Project Status -----------------------------------------------------------
section_header("Project Status")
st.info(
    "All components of the system — preprocessing, model training, SHAP explainability, "
    "business-scenario validation, and this decision-support interface — are complete and "
    "connected through a dedicated service layer."
)

elapsed = time.perf_counter() - _page_start
logger.info("About page loaded in %.3fs.", elapsed)

footer()
