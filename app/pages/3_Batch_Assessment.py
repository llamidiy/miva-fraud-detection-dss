"""Batch Assessment — score a CSV of transactions at once.

Phase 2: an uploaded CSV is scored through `prediction_service.predict_batch`
(which reuses the frozen predictor — no per-row SHAP, for practicality at
batch scale).

Phase 3 usability fix: a downloadable CSV template and clearer upload
validation, built from the predictor's own required-column contract so
the template can never drift out of sync with what the predictor
actually requires.

Sprint 6.3.5 — UX & traceability refinement:

- `isFlaggedFraud` is no longer part of the user-facing schema. The
  model was trained on it and still receives it — `predict_batch`
  already defaults it to 0 whenever it's absent from the uploaded
  frame (see `_REQUIRED_BATCH_COLUMNS` in `prediction_service`), which
  is exactly the "app supplies isFlaggedFraud=0" abstraction this
  sprint asks for. This page achieves it by simply never forwarding
  that column to the predictor — no service-layer change needed.
- Upload no longer scores immediately: the file is read, validated,
  and previewed first; scoring only happens once the analyst clicks
  "Assess Transactions".
- Every uploaded row gets a stable `assessment_id` (``TXN-0001``, ...)
  assigned once at preview time and carried through to the results
  table and the downloaded CSV, so a result can always be traced back
  to the row that produced it.
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

from components.cards import info_box, metric_card
from components.layout import footer, page_title, section_header
from components.tables import data_table, placeholder_table
from services.prediction_service import _REQUIRED_BATCH_COLUMNS, predict_batch

logger = logging.getLogger(__name__)

#: The transaction fields an analyst must provide. Sourced directly from
#: the predictor's own contract in `services.prediction_service` — this
#: already excludes `isFlaggedFraud` (see that module's docstring), so
#: no fraud-related field is ever requested from the user.
USER_FACING_FIELDS: list[str] = list(_REQUIRED_BATCH_COLUMNS)

#: Presentational metadata only (type/description/example) — the column
#: names and order come from `USER_FACING_FIELDS` above, not from this
#: dict, so the displayed schema can't drift from the real one.
_COLUMN_HELP: dict[str, tuple[str, str, str]] = {
    "step": ("integer", "Simulated time step (hours since the start of the simulation, 1-based).", "1"),
    "type": ("string", "Transaction type: one of CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER.", "PAYMENT"),
    "amount": ("float", "Transaction amount.", "9839.64"),
    "oldbalanceOrg": ("float", "Origin account balance before the transaction.", "170136.00"),
    "newbalanceOrig": ("float", "Origin account balance after the transaction.", "160296.36"),
    "oldbalanceDest": ("float", "Destination account balance before the transaction.", "0.00"),
    "newbalanceDest": ("float", "Destination account balance after the transaction.", "0.00"),
}

_RESULT_COLUMNS = ["assessment_id", *USER_FACING_FIELDS, "prediction", "confidence", "recommendation"]

_page_start = time.perf_counter()

page_title(
    "Batch Assessment",
    subtitle="Upload a CSV of transactions to assess them all at once.",
)

# --- Download CSV Template -----------------------------------------------------
section_header(
    "Download CSV Template",
    "Use this template to prepare transaction data for batch assessment. The system "
    "automatically handles feature engineering and internal model fields.",
)
template_df = pd.DataFrame(columns=USER_FACING_FIELDS)
template_clicked = st.download_button(
    "Download Batch Assessment Template",
    data=template_df.to_csv(index=False).encode("utf-8"),
    file_name="batch_assessment_template.csv",
    mime="text/csv",
)
if template_clicked:
    logger.info("Batch Assessment CSV template downloaded.")

with st.expander("View Required Columns"):
    column_rows = [
        {
            "Column": col,
            "Type": _COLUMN_HELP[col][0],
            "Description": _COLUMN_HELP[col][1],
            "Example": _COLUMN_HELP[col][2],
        }
        for col in USER_FACING_FIELDS
    ]
    data_table(column_rows)
    st.caption("Example row (for reference only — the downloaded template is blank):")
    data_table([{col: _COLUMN_HELP[col][2] for col in USER_FACING_FIELDS}])

# --- Upload Transactions ---------------------------------------------------------
section_header("Upload Transactions", "Accepts a CSV file matching the transaction fields above.")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if "batch_file_signature" not in st.session_state:
    st.session_state.batch_file_signature = None
    st.session_state.batch_working_df = None
    st.session_state.batch_missing = None
    st.session_state.batch_extra = None
    st.session_state.batch_assessment = None

if uploaded_file is None:
    st.session_state.batch_file_signature = None
    st.session_state.batch_working_df = None
    st.session_state.batch_missing = None
    st.session_state.batch_extra = None
    st.session_state.batch_assessment = None
else:
    signature = f"{uploaded_file.name}:{uploaded_file.size}"
    if signature != st.session_state.batch_file_signature:
        # A genuinely new upload — (re)validate and build the preview.
        # Any prior assessment belongs to the old file, so it's cleared.
        st.session_state.batch_file_signature = signature
        st.session_state.batch_assessment = None
        try:
            df_raw = pd.read_csv(uploaded_file)
        except Exception as exc:
            logger.error("Batch Assessment could not read %s: %s", uploaded_file.name, exc)
            st.session_state.batch_working_df = None
            st.session_state.batch_missing = None
            st.session_state.batch_extra = None
            st.session_state.batch_read_error = str(exc)
        else:
            st.session_state.batch_read_error = None
            logger.info("Batch Assessment CSV uploaded: %s (%d rows).", uploaded_file.name, len(df_raw))
            missing = [c for c in USER_FACING_FIELDS if c not in df_raw.columns]
            if missing:
                logger.warning(
                    "Batch Assessment upload rejected: %s is missing required column(s) %s.",
                    uploaded_file.name,
                    missing,
                )
                st.session_state.batch_working_df = None
                st.session_state.batch_missing = missing
                st.session_state.batch_extra = None
            else:
                working_df = df_raw.reset_index(drop=True).copy()
                working_df.insert(0, "assessment_id", [f"TXN-{i + 1:04d}" for i in range(len(working_df))])
                st.session_state.batch_working_df = working_df
                st.session_state.batch_missing = None
                st.session_state.batch_extra = [c for c in df_raw.columns if c not in USER_FACING_FIELDS]
                st.session_state.batch_filename = uploaded_file.name
                logger.info(
                    "Batch Assessment preview generated for %s: %d record(s), %d column(s) provided.",
                    uploaded_file.name,
                    len(working_df),
                    len(df_raw.columns),
                )

# --- Uploaded File Preview -------------------------------------------------------
section_header("Uploaded File Preview", "What was uploaded, before anything is assessed.")

if uploaded_file is None:
    info_box(
        f"No file uploaded yet. Required columns: {', '.join(USER_FACING_FIELDS)}. "
        "Use the template above to get started.",
        kind="neutral",
    )
    placeholder_table(
        columns=_RESULT_COLUMNS,
        caption="Assessment results will appear here once a file is uploaded and assessed.",
    )
elif st.session_state.get("batch_read_error"):
    info_box(f"Could not read `{uploaded_file.name}`: {st.session_state.batch_read_error}", kind="danger")
elif st.session_state.batch_missing:
    info_box(
        f"Cannot assess `{uploaded_file.name}` — missing required column(s): "
        f"<strong>{', '.join(st.session_state.batch_missing)}</strong>. Download the CSV template above, "
        "add your transaction data, and try again.",
        kind="danger",
    )
else:
    working_df = st.session_state.batch_working_df
    filename = st.session_state.batch_filename
    n_records = len(working_df)
    n_columns = len(working_df.columns) - 1  # exclude the app-added assessment_id

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("File Name", filename)
    with col2:
        metric_card("Records", f"{n_records:,}")
    with col3:
        metric_card("Columns Provided", str(n_columns))
    with col4:
        metric_card("Validation Status", "Valid")

    if st.session_state.batch_extra:
        info_box(
            "Note: the following uploaded column(s) are not used by the model and will be ignored "
            f"during assessment: {', '.join(st.session_state.batch_extra)}.",
            kind="warning",
        )

    data_table(working_df.to_dict("records"), caption="Every uploaded record, exactly as provided.")

    assess_clicked = st.button("Assess Transactions", type="primary")

    if assess_clicked:
        start = time.perf_counter()
        logger.info("Batch Assessment assessment initiated for %s: %d transaction(s).", filename, n_records)

        df_for_prediction = working_df[USER_FACING_FIELDS].copy()
        with st.spinner(f"Assessing {n_records:,} transaction(s)..."):
            batch_result = predict_batch(df_for_prediction)

        if batch_result.errors:
            logger.error(
                "Batch Assessment assessment failed for %s: %s", filename, "; ".join(batch_result.errors)
            )
            st.session_state.batch_assessment = {"error": "; ".join(batch_result.errors)}
        else:
            results_df = working_df[["assessment_id", *USER_FACING_FIELDS]].copy()
            results_df["prediction"] = [r.prediction for r in batch_result.results]
            results_df["confidence"] = [r.confidence for r in batch_result.results]
            results_df["recommendation"] = [r.recommendation for r in batch_result.results]

            n_fraud = int((results_df["prediction"] == "Fraud").sum())
            n_not_fraud = len(results_df) - n_fraud
            completion_pct = (batch_result.n_processed / n_records * 100) if n_records else 0.0
            elapsed = time.perf_counter() - start

            st.session_state.batch_assessment = {
                "error": None,
                "results_df": results_df,
                "filename": filename,
                "n_total": n_records,
                "n_processed": batch_result.n_processed,
                "n_fraud": n_fraud,
                "n_not_fraud": n_not_fraud,
                "completion_pct": completion_pct,
            }
            logger.info(
                "Batch Assessment assessment completed for %s in %.3fs: %d processed, %d fraud, %d not fraud.",
                filename,
                elapsed,
                batch_result.n_processed,
                n_fraud,
                n_not_fraud,
            )

# --- Assessment Summary & Results -------------------------------------------------
assessment = st.session_state.batch_assessment
if assessment is not None:
    if assessment["error"]:
        section_header("Assessment Summary", "")
        info_box(assessment["error"], kind="danger")
    else:
        section_header("Assessment Summary", "Assessment Complete.")
        info_box(
            f"Assessment Complete — {assessment['n_total']:,} transaction(s) assessed from "
            f"`{assessment['filename']}`.",
            kind="success",
        )
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            metric_card("Transactions Assessed", f"{assessment['n_total']:,}")
        with col2:
            metric_card("Fraud Flagged", f"{assessment['n_fraud']:,}")
        with col3:
            metric_card("Not Fraud", f"{assessment['n_not_fraud']:,}")
        with col4:
            metric_card("Completion Status", f"{assessment['completion_pct']:.0f}% Completed")

        section_header(
            "Detailed Results",
            "Assessment ID, the input fields that were assessed, and the model's output for each transaction.",
        )
        results_df = assessment["results_df"]
        data_table(results_df.to_dict("records"), caption="Match the Assessment ID to trace a result back to its input row.")

        download_clicked = st.download_button(
            "Download Results (CSV)",
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name=f"assessment_{Path(assessment['filename']).stem}.csv",
            mime="text/csv",
        )
        if download_clicked:
            logger.info("Batch Assessment results downloaded for %s.", assessment["filename"])

_elapsed = time.perf_counter() - _page_start
logger.info("Batch Assessment page loaded in %.3fs.", _elapsed)

footer()
