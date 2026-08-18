"""Validation — business scenario validation results.

Phase 3: fully wired to Sprint 5.5's validation outputs through
`services.validation_service` and `services.report_service`. No
scenario is re-run and no SHAP value is recomputed on this page.
"""

import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parents[1]
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import streamlit as st

from components.cards import info_box, metric_card, status_badge, status_badge_html
from components.charts import chart_placeholder
from components.layout import footer, page_title, section_header
from components.tables import data_table
from services.report_service import get_business_scenarios_report, get_validation_report
from state import get_app_state
from utils.paths import get_scenario_waterfall_path

logger = logging.getLogger(__name__)

_page_start = time.perf_counter()
logger.info("Loading Validation page...")

page_title(
    "Validation",
    subtitle="Business scenario validation results: fraud, legitimate, and edge cases.",
)

app_state = get_app_state()
summary = app_state.validation_summary
logger.info("Loaded validation summary: %d scenarios, error=%s.", summary.total_scenarios, summary.error)

if summary.error:
    info_box(f"Validation results are not available: {summary.error}", kind="danger")
    footer()
    st.stop()

info_box(
    "These scenarios test whether the complete system — prediction, explanation, and "
    "recommendation — behaves sensibly across representative fraud, legitimate, and edge-case "
    "transaction patterns. Two things are measured separately: <strong>PASS</strong> means a "
    "scenario executed end-to-end without error; <strong>Match</strong> means the model's "
    "prediction agreed with the scenario's expected outcome. A scenario can execute successfully "
    "and still mismatch — mismatches are reported openly below.",
    kind="info",
)

# --- Validation Overview -----------------------------------------------------
section_header("Validation Overview", "Aggregate outcome across all validated scenarios.")
col1, col2, col3, col4 = st.columns(4)
with col1:
    metric_card("Total Scenarios", str(summary.total_scenarios))
with col2:
    pass_rate = (summary.passed / summary.total_scenarios * 100) if summary.total_scenarios else 0.0
    metric_card("Pass Rate", f"{pass_rate:.0f}%", help_text=f"{summary.passed}/{summary.total_scenarios} PASS")
with col3:
    metric_card("Expected vs. Actual Matches", str(summary.matches))
with col4:
    metric_card("Mismatches", str(summary.mismatches), help_text=f"{summary.exploratory} exploratory (no strong prior)")

# --- Scenario Summary Cards (by category) ------------------------------------
section_header("Scenario Summary", "Coverage and match rate by scenario category.")
by_category = defaultdict(lambda: {"total": 0, "match": 0})
for row in summary.scenarios:
    by_category[row["category"]]["total"] += 1
    if row["outcome_match"] == "Match":
        by_category[row["category"]]["match"] += 1

category_labels = {"fraud": "Fraud Scenarios", "legitimate": "Legitimate Scenarios", "edge_case": "Edge Cases"}
cols = st.columns(len(category_labels))
for col, (key, label) in zip(cols, category_labels.items()):
    stats = by_category.get(key, {"total": 0, "match": 0})
    with col:
        metric_card(label, str(stats["total"]), help_text=f"{stats['match']} matched expectation")

# --- Interactive Scenario Table -----------------------------------------------
section_header("Interactive Scenario Table", "All scenarios — sortable and searchable.")
table_rows = [
    {
        "Scenario ID": row["scenario_id"],
        "Title": row["title"],
        "Category": row["category"],
        "Prediction": row["prediction"],
        "Confidence": row["confidence_pct"],
        "Outcome Match": row["outcome_match"],
        "Validation Status": row["validation_status"],
    }
    for row in summary.scenarios
]
data_table(table_rows, caption="Click a column header to sort. Use the search icon to filter.")

# --- Scenario Details ----------------------------------------------------------
section_header("Scenario Details", "Select a scenario to see its full explanation and recommendation.")
scenario_ids = [row["scenario_id"] for row in summary.scenarios]
selected_id = st.selectbox("Scenario", options=scenario_ids)
selected = next((row for row in summary.scenarios if row["scenario_id"] == selected_id), None)
logger.info("Scenario detail selected: %s", selected_id)

if selected is None:
    info_box("No scenario selected.", kind="neutral")
else:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(f"**{selected['title']}**")
        status_badge(selected["category"], kind="info")
        st.write("")
        match_kind = {"Match": "success", "Mismatch": "danger"}.get(selected["outcome_match"], "neutral")
        st.markdown(
            f"Prediction: **{selected['prediction']}** ({selected['confidence_pct']}% confidence) "
            + status_badge_html(selected["outcome_match"], kind=match_kind),
            unsafe_allow_html=True,
        )
        st.markdown("**SHAP-Based Contributing Factors**")
        for factor in str(selected.get("top_shap_factors", "")).split(";"):
            factor = factor.strip()
            if factor:
                st.markdown(f"- {factor}")
        st.markdown(f"**Recommendation:** {selected.get('recommendation_actions', 'N/A')}")
        status_kind = "success" if selected["validation_status"] == "PASS" else "danger"
        st.markdown("**Validation Status:** " + status_badge_html(selected["validation_status"], kind=status_kind), unsafe_allow_html=True)

    with col2:
        waterfall_path = get_scenario_waterfall_path(selected_id)
        if waterfall_path.exists():
            st.image(str(waterfall_path), caption=f"SHAP Waterfall — {selected_id}", use_container_width=True)
        else:
            chart_placeholder("Waterfall plot unavailable", size="md")

# --- Full Reports (reused, not regenerated) ------------------------------------
section_header("Full Reports", "The complete validation documents generated by the scenario run.")
validation_report = get_validation_report()
business_scenarios = get_business_scenarios_report()
logger.info(
    "Loaded full report documents: validation_report=%s, business_scenarios=%s.",
    validation_report.available,
    business_scenarios.available,
)

with st.expander("Full Validation Report"):
    if validation_report.available:
        st.markdown(validation_report.content)
    else:
        info_box("Validation report is not available.", kind="warning")

with st.expander("Business Scenario Catalog"):
    if business_scenarios.available:
        st.markdown(business_scenarios.content)
    else:
        info_box("Business scenario catalog is not available.", kind="warning")

elapsed = time.perf_counter() - _page_start
logger.info("Validation page loaded in %.3fs.", elapsed)

footer()
