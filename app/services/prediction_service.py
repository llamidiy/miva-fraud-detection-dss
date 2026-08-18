"""Prediction service.

UI-facing wrapper around the frozen predictor and explainability layer.
Never returns raw backend objects (DataFrames, `shap.Explanation`, the
`FraudExplainer` itself) — only simple, UI-friendly dataclasses.

`predict_transaction` reuses `explainability_service.get_local_explanation`
(a full SHAP-based explanation) since the Single Transaction page needs
prediction, confidence, recommendation, factors, narrative, and a
waterfall plot together. `predict_batch` deliberately skips per-row SHAP
computation — generating a full explanation for every row of an
uploaded file would be impractically slow — and only computes
predictions, using the same frozen predictor.
"""

import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

_APP_DIR = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _APP_DIR.parent
for _path in (_APP_DIR, _PROJECT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pandas as pd

from src.preprocessing.feature_engineering import engineer_features
from src.validation.recommendation_engine import generate_recommendation
from src.validation.scenario_generator import RAW_TRANSACTION_FIELDS

from services.explainability_service import get_champion_explainer, get_local_explanation

logger = logging.getLogger(__name__)

#: Raw transaction fields required for a batch upload (isFlaggedFraud is
#: optional and defaults to 0 if the uploaded file omits it).
_REQUIRED_BATCH_COLUMNS = [f for f in RAW_TRANSACTION_FIELDS if f != "isFlaggedFraud"]


@dataclass
class PredictionResult:
    """A single transaction's prediction, explanation, and recommendation.

    Attributes:
        prediction: ``"Fraud"`` or ``"Not Fraud"``, or ``"N/A"`` on failure.
        confidence: Confidence in `prediction`, as a percentage.
        recommendation: A display-ready, comma-separated list of
            recommended actions.
        top_factors: Ranked, human-readable SHAP contributing factors.
        waterfall_path: Path to the saved waterfall plot.
        narrative: The dashboard-ready natural-language explanation.
        error: Set if the request failed; other fields hold safe defaults.
    """

    prediction: str
    confidence: float
    recommendation: str
    top_factors: list[str]
    waterfall_path: str
    narrative: str
    error: Optional[str] = None


def predict_transaction(transaction: dict[str, Any]) -> PredictionResult:
    """Predict, explain, and recommend an action for one ad-hoc transaction.

    Reuses `explainability_service.get_local_explanation`, which already
    wraps the frozen predictor and SHAP explanation logic end to end —
    no prediction or SHAP logic is reimplemented here.

    Args:
        transaction: Raw transaction fields (step, type, amount,
            oldbalanceOrg, newbalanceOrig, oldbalanceDest,
            newbalanceDest, and optionally isFlaggedFraud, which
            defaults to 0).

    Returns:
        A `PredictionResult`. On failure, `error` is set and other
        fields hold safe defaults.
    """
    start = time.perf_counter()
    full_transaction = {**transaction, "isFlaggedFraud": transaction.get("isFlaggedFraud", 0)}

    explanation = get_local_explanation(full_transaction)
    elapsed = time.perf_counter() - start

    if explanation.error:
        logger.error("Prediction request failed after %.3fs: %s", elapsed, explanation.error)
        return PredictionResult(
            prediction="N/A",
            confidence=0.0,
            recommendation="N/A",
            top_factors=[],
            waterfall_path="",
            narrative="",
            error=explanation.error,
        )

    logger.info(
        "Prediction request completed in %.3fs: prediction=%s confidence=%.1f%%",
        elapsed,
        explanation.prediction,
        explanation.confidence,
    )
    return PredictionResult(
        prediction=explanation.prediction,
        confidence=explanation.confidence,
        recommendation=", ".join(explanation.recommendation_actions),
        top_factors=explanation.top_factors,
        waterfall_path=explanation.waterfall_path,
        narrative=explanation.narrative,
    )


@dataclass
class BatchTransactionResult:
    """One row's prediction within a batch request."""

    row_index: int
    prediction: str
    confidence: float
    recommendation: str


@dataclass
class BatchPredictionResult:
    """The outcome of a batch prediction request.

    Attributes:
        results: One entry per successfully scored row.
        n_processed: Number of rows successfully scored.
        n_failed: Number of rows that could not be scored.
        errors: Any error messages encountered.
    """

    results: list[BatchTransactionResult] = field(default_factory=list)
    n_processed: int = 0
    n_failed: int = 0
    errors: list[str] = field(default_factory=list)


def predict_batch(df_raw: pd.DataFrame) -> BatchPredictionResult:
    """Predict and recommend an action for every row in an uploaded transaction table.

    Computes predictions only (no per-row SHAP explanation, which would
    be impractical at batch scale) via the same frozen predictor used by
    `predict_transaction`.

    Args:
        df_raw: A DataFrame with the required raw transaction columns
            (see `_REQUIRED_BATCH_COLUMNS`). `isFlaggedFraud` defaults
            to 0 if not present.

    Returns:
        A `BatchPredictionResult` with one entry per successfully scored
        row, and any errors encountered.
    """
    start = time.perf_counter()

    missing = [c for c in _REQUIRED_BATCH_COLUMNS if c not in df_raw.columns]
    if missing:
        error = f"Uploaded file is missing required columns: {missing}"
        logger.error("Batch prediction request rejected: %s", error)
        return BatchPredictionResult(n_failed=len(df_raw), errors=[error])

    if df_raw.empty:
        error = "Uploaded file contains no rows."
        logger.error("Batch prediction request rejected: %s", error)
        return BatchPredictionResult(errors=[error])

    df = df_raw.copy()
    if "isFlaggedFraud" not in df.columns:
        df["isFlaggedFraud"] = 0

    try:
        engineered = engineer_features(df[[*RAW_TRANSACTION_FIELDS]])
        explainer = get_champion_explainer()
        outcome = explainer.predict_transaction(engineered)
        predictions = outcome["prediction"]
        probabilities = outcome["fraud_probability"]

        results: list[BatchTransactionResult] = []
        for i in range(len(engineered)):
            label = "Fraud" if int(predictions[i]) == 1 else "Not Fraud"
            probability = float(probabilities[i]) if probabilities is not None else 0.0
            confidence = (probability if label == "Fraud" else 1 - probability) * 100
            recommendation = generate_recommendation(label, confidence)
            results.append(
                BatchTransactionResult(
                    row_index=i,
                    prediction=label,
                    confidence=round(confidence, 2),
                    recommendation=", ".join(recommendation.actions),
                )
            )

        elapsed = time.perf_counter() - start
        logger.info("Batch prediction request completed in %.3fs: %d rows scored.", elapsed, len(results))
        return BatchPredictionResult(results=results, n_processed=len(results))

    except Exception as exc:
        elapsed = time.perf_counter() - start
        logger.exception("Batch prediction request failed after %.3fs.", elapsed)
        return BatchPredictionResult(n_failed=len(df_raw), errors=[str(exc)])
