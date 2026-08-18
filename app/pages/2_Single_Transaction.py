"""Single Transaction — assess one transaction interactively.

The core analyst workflow: input → assess → prediction → confidence →
recommendation → explanation. Submitting the form runs the champion
model end to end through `prediction_service.predict_transaction`,
which reuses the frozen predictor and SHAP explanation logic — no
prediction or SHAP logic is reimplemented on this page.
"""

import logging
import sys
import time
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parents[1]
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import streamlit as st

from components.cards import info_box, metric_card, placeholder_card, status_badge_html
from components.charts import chart_placeholder
from components.forms import transaction_input_form
from components.layout import footer, page_title, section_header
from services.prediction_service import predict_transaction

logger = logging.getLogger(__name__)

_page_start = time.perf_counter()
logger.info("Loading Single Transaction page...")

page_title(
    "Single Transaction Assessment",
    subtitle="Enter a transaction's details to get a prediction, a recommended action, and an explanation.",
)

section_header("Transaction Details", "The raw transaction fields the model expects.")
form_values = transaction_input_form()

if form_values is not None:
    transaction = {
        "step": form_values["step"],
        "type": form_values["transaction_type"],
        "amount": form_values["amount"],
        "oldbalanceOrg": form_values["old_balance_org"],
        "newbalanceOrig": form_values["new_balance_orig"],
        "oldbalanceDest": form_values["old_balance_dest"],
        "newbalanceDest": form_values["new_balance_dest"],
    }
    logger.info("Single transaction assessment submitted (type=%s).", form_values["transaction_type"])
    with st.spinner("Assessing transaction..."):
        st.session_state["last_prediction_result"] = predict_transaction(transaction)

result = st.session_state.get("last_prediction_result")

section_header("Prediction", "The model's decision, its confidence, and the recommended analyst action.")
if result is None:
    placeholder_card("Prediction", "Submit a transaction above to see the model's decision.")
elif result.error:
    info_box(
        f"The assessment could not be completed: {result.error} "
        "Check the transaction values and try again.",
        kind="danger",
    )
else:
    pred_kind = "danger" if result.prediction == "Fraud" else "success"
    col1, col2 = st.columns(2)
    with col1:
        metric_card("Prediction", status_badge_html(result.prediction, kind=pred_kind))
    with col2:
        metric_card(
            "Confidence",
            f"{result.confidence:.1f}%",
            help_text="How strongly the model supports this prediction",
        )
    info_box(f"<strong>Recommendation:</strong> {result.recommendation}", kind=pred_kind)

section_header(
    "Why This Decision?",
    "The features that contributed most to this prediction, estimated with SHAP. "
    "Contributions describe how the model reasoned — they are not proof that a feature caused fraud.",
)
if result is None:
    col1, col2 = st.columns(2)
    with col1:
        chart_placeholder("The SHAP waterfall plot will appear here after an assessment", size="md")
    with col2:
        placeholder_card(
            "Top Contributing Factors",
            "Submit a transaction above to see which features contributed to the prediction.",
        )
elif result.error:
    info_box("Explanation unavailable — the assessment above did not complete.", kind="neutral")
else:
    col1, col2 = st.columns(2)
    with col1:
        if result.waterfall_path and Path(result.waterfall_path).exists():
            st.image(
                result.waterfall_path,
                caption="SHAP waterfall — how each feature pushed this prediction from the model's baseline to its final decision.",
                use_container_width=True,
            )
        else:
            chart_placeholder("Waterfall plot unavailable for this assessment", size="md")
    with col2:
        with st.container(border=True):
            st.markdown("**Top Contributing Factors**")
            for factor in result.top_factors:
                st.markdown(f"- {factor}")
        st.caption(
            "Factors are ranked by the size of their SHAP contribution to this specific prediction."
        )

    with st.expander("Copyable case summary"):
        st.caption("A plain-text summary of this assessment, ready to paste into case notes.")
        st.code(result.narrative, language=None)

elapsed = time.perf_counter() - _page_start
logger.info("Single Transaction page loaded in %.3fs.", elapsed)

footer()
