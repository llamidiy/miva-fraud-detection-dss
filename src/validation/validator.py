"""Runs each scenario through the full decision-support pipeline.

For every scenario:

    Scenario -> Prediction -> Confidence -> SHAP Explanation -> Recommendation

Reuses the frozen Sprint 5 explainability layer
(`src.explainability.local_explanations.explain_single_transaction`) and
the frozen Sprint 3 feature engineering (`engineer_features`) exactly as
they are — nothing here recomputes SHAP logic or engineered-feature
formulas independently. Any runtime failure at any stage is caught and
recorded on the result rather than aborting the run.
"""

import logging
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from src.config import VALIDATION_SCREENSHOTS_DIR
from src.explainability.explainer import FraudExplainer
from src.explainability.local_explanations import explain_single_transaction
from src.preprocessing.feature_engineering import engineer_features
from src.validation.recommendation_engine import Recommendation, generate_recommendation
from src.validation.scenario_generator import RAW_TRANSACTION_FIELDS, Scenario

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """The outcome of running one scenario through the full pipeline.

    Attributes:
        scenario: The scenario that was validated.
        prediction: ``"Fraud"`` / ``"Not Fraud"``, or ``"N/A"`` if the
            pipeline failed before a prediction was produced.
        confidence_pct: Confidence in `prediction`, as a percentage.
        top_shap_factors: Ranked, human-readable SHAP contributing factors.
        narrative: The dashboard-style natural-language explanation.
        waterfall_plot_path: Path to this scenario's saved waterfall plot.
        recommendation: The `Recommendation` produced by the
            recommendation engine, or `None` if the pipeline failed
            before a recommendation could be generated.
        outcome_match: ``"Match"``, ``"Mismatch"``, or an ``"N/A (...)"``
            explanation, comparing `scenario.expected_outcome` against
            `recommendation.tier`.
        prediction_succeeded: Whether a valid prediction was produced.
        explanation_succeeded: Whether at least one SHAP factor was produced.
        recommendation_succeeded: Whether a recommendation was produced.
        confidence_exists: Whether a numeric confidence score was produced.
        validation_status: ``"PASS"`` if every check above succeeded and
            no error occurred, else ``"FAIL"``.
        error_message: The exception message, if the pipeline raised.
    """

    scenario: Scenario
    prediction: str
    confidence_pct: float
    top_shap_factors: list[str]
    narrative: str
    waterfall_plot_path: str
    recommendation: Optional[Recommendation]
    outcome_match: str
    prediction_succeeded: bool
    explanation_succeeded: bool
    recommendation_succeeded: bool
    confidence_exists: bool
    validation_status: str
    error_message: Optional[str] = None


def build_transaction_dataframe(transaction: dict[str, Any]) -> pd.DataFrame:
    """Build a single-row, fully engineered transaction DataFrame.

    Takes only the raw PaySim-style fields (see
    `src.validation.scenario_generator.RAW_TRANSACTION_FIELDS`) and
    derives the engineered columns (balanceDeltaOrig, errorBalanceOrig,
    logAmount, transactionHour, ...) via the frozen `engineer_features`
    function, so scenario data is transformed identically to how the
    model was trained rather than hand-computed.

    Args:
        transaction: A dict with keys `RAW_TRANSACTION_FIELDS`.

    Returns:
        A single-row DataFrame with all raw and engineered columns.
    """
    base = {field_name: [transaction[field_name]] for field_name in RAW_TRANSACTION_FIELDS}
    return engineer_features(pd.DataFrame(base))


def validate_scenario(
    explainer: FraudExplainer, scenario: Scenario, reference_sample: pd.DataFrame
) -> ValidationResult:
    """Run one scenario through prediction, explanation, and recommendation.

    Args:
        explainer: An initialized `FraudExplainer` (reused across scenarios).
        scenario: The scenario to validate.
        reference_sample: A representative sample (raw columns) used to
            compute "High"/"Low" reference statistics for the local
            explanation — reuses Sprint 5's `shap_sample.csv`.

    Returns:
        The `ValidationResult` for this scenario. Never raises: any
        exception during the pipeline is caught and recorded on the
        result instead.
    """
    prediction = "N/A"
    confidence_pct = float("nan")
    top_factors: list[str] = []
    narrative = ""
    waterfall_path = ""
    recommendation: Optional[Recommendation] = None
    error_message: Optional[str] = None

    try:
        transaction_df = build_transaction_dataframe(scenario.transaction)

        local_result = explain_single_transaction(
            explainer,
            transaction_df,
            reference_sample=reference_sample,
            waterfall_path=VALIDATION_SCREENSHOTS_DIR / f"waterfall_{scenario.scenario_id}.png",
        )
        prediction = local_result["prediction"]
        confidence_pct = local_result["confidence_pct"]
        top_factors = local_result["top_factors"]
        narrative = local_result["narrative"]
        waterfall_path = local_result["waterfall_plot_path"]

        recommendation = generate_recommendation(prediction, confidence_pct)

    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        logger.exception("Scenario '%s' failed during validation.", scenario.scenario_id)

    prediction_succeeded = prediction in {"Fraud", "Not Fraud"}
    confidence_exists = confidence_pct is not None and confidence_pct == confidence_pct  # not NaN
    explanation_succeeded = len(top_factors) > 0
    recommendation_succeeded = recommendation is not None and bool(recommendation.actions)

    validation_status = (
        "PASS"
        if all(
            [
                prediction_succeeded,
                explanation_succeeded,
                recommendation_succeeded,
                confidence_exists,
                error_message is None,
            ]
        )
        else "FAIL"
    )

    if scenario.expected_outcome == "uncertain":
        outcome_match = "N/A (no strong prior expectation)"
    elif recommendation is None:
        outcome_match = "N/A (validation failed)"
    else:
        outcome_match = "Match" if recommendation.tier == scenario.expected_outcome else "Mismatch"

    logger.info(
        "Validated '%s' (%s): status=%s prediction=%s confidence=%s outcome_match=%s",
        scenario.scenario_id,
        scenario.title,
        validation_status,
        prediction,
        f"{confidence_pct:.1f}%" if confidence_exists else "N/A",
        outcome_match,
    )

    return ValidationResult(
        scenario=scenario,
        prediction=prediction,
        confidence_pct=confidence_pct,
        top_shap_factors=top_factors,
        narrative=narrative,
        waterfall_plot_path=waterfall_path,
        recommendation=recommendation,
        outcome_match=outcome_match,
        prediction_succeeded=prediction_succeeded,
        explanation_succeeded=explanation_succeeded,
        recommendation_succeeded=recommendation_succeeded,
        confidence_exists=confidence_exists,
        validation_status=validation_status,
        error_message=error_message,
    )
