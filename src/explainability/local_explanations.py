"""Local (single-transaction) explanations for the champion fraud model.

Produces a full decision-support explanation for one raw transaction:
its prediction, confidence, ranked SHAP contributing factors, a
waterfall plot, and a concise natural-language summary suitable for a
dashboard.
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import FIGURES_DIR
from src.explainability import visualization
from src.explainability.explainer import FraudExplainer

logger = logging.getLogger(__name__)

#: Human-readable descriptions for engineered/raw feature columns.
FEATURE_DESCRIPTIONS: dict[str, str] = {
    "step": "transaction time step",
    "amount": "transaction amount",
    "oldbalanceOrg": "origin account's starting balance",
    "newbalanceOrig": "origin account's ending balance",
    "oldbalanceDest": "destination account's starting balance",
    "newbalanceDest": "destination account's ending balance",
    "isFlaggedFraud": "flagged by the existing rule-based system",
    "balanceDeltaOrig": "change in origin account balance",
    "balanceDeltaDest": "change in destination account balance",
    "errorBalanceOrig": "balance discrepancy on the origin account",
    "errorBalanceDest": "balance discrepancy on the destination account",
    "isOrigZeroBalance": "zero origin balance",
    "isDestZeroBalance": "zero destination balance",
    "logAmount": "transaction amount (log-scaled)",
    "transactionDay": "day of the transaction",
    "transactionHour": "hour of the transaction",
    "type_CASH_IN": "transaction type is cash-in",
    "type_CASH_OUT": "transaction type is cash-out",
    "type_DEBIT": "transaction type is debit",
    "type_PAYMENT": "transaction type is payment",
    "type_TRANSFER": "transaction type is transfer",
}

#: Feature name prefixes treated as boolean flags rather than continuous values.
_BOOLEAN_LIKE_PREFIXES: tuple[str, ...] = ("is", "type_")

#: Default destination for the single-transaction waterfall plot.
_DEFAULT_WATERFALL_PATH = FIGURES_DIR / "waterfall_example.png"


def _describe_factor(feature_name: str, raw_value: float, reference_median: float) -> str:
    """Build a short, human-readable description of one contributing factor.

    Args:
        feature_name: The encoded feature's column name.
        raw_value: The feature's (encoded) value for this transaction.
        reference_median: The feature's median value across the
            representative sample, used to label continuous features as
            "High"/"Low".

    Returns:
        A short phrase, e.g. ``"High transaction amount"`` or ``"Zero
        origin balance"``.
    """
    description = FEATURE_DESCRIPTIONS.get(feature_name, feature_name)

    if feature_name.startswith(_BOOLEAN_LIKE_PREFIXES):
        if raw_value >= 0.5:
            return description[0].upper() + description[1:]
        return f"Not {description}"

    level = "High" if raw_value >= reference_median else "Low"
    return f"{level} {description}"


def _top_contributing_factors(
    explanation_single: Any, reference_medians: pd.Series, top_n: int = 3
) -> list[str]:
    """Rank a single transaction's SHAP contributions and describe the top factors.

    Args:
        explanation_single: A single-row `shap.Explanation`.
        reference_medians: Per-feature medians from the representative
            sample (encoded), used to label continuous features.
        top_n: Number of top factors to return.

    Returns:
        A list of short, human-readable factor descriptions, ranked by
        absolute SHAP impact (most impactful first).
    """
    feature_names = list(explanation_single.feature_names)
    values = explanation_single.values
    data = explanation_single.data

    ranked = sorted(zip(feature_names, values, data), key=lambda item: abs(item[1]), reverse=True)

    factors = []
    for feature_name, _shap_value, raw_value in ranked[:top_n]:
        median = reference_medians.get(feature_name, 0.0)
        factors.append(_describe_factor(feature_name, raw_value, median))
    return factors


def _build_narrative(
    prediction_label: str, confidence_pct: float, factors: list[str], recommendation: str
) -> str:
    """Assemble the full concise natural-language explanation block.

    Args:
        prediction_label: ``"Fraud"`` or ``"Not Fraud"``.
        confidence_pct: Confidence in the predicted label, as a percentage.
        factors: Ranked, human-readable contributing factors.
        recommendation: A short recommended action.

    Returns:
        A multi-line, dashboard-ready explanation string.
    """
    factor_lines = "\n".join(f"• {factor}" for factor in factors)
    return (
        f"Prediction: {prediction_label}\n"
        f"Confidence: {confidence_pct:.1f}%\n\n"
        f"Top contributing factors\n"
        f"{factor_lines}\n\n"
        f"Recommendation\n"
        f"{recommendation}"
    )


def explain_single_transaction(
    explainer: FraudExplainer,
    transaction: pd.DataFrame,
    reference_sample: pd.DataFrame,
    waterfall_path: Path = _DEFAULT_WATERFALL_PATH,
    top_n: int = 3,
) -> dict[str, Any]:
    """Produce a full local explanation for one raw transaction.

    Args:
        explainer: An initialized `FraudExplainer`.
        transaction: A single-row DataFrame of raw (pre-encoding)
            transaction data.
        reference_sample: The representative sample (raw columns; the
            target column, if present, is ignored), used to compute
            reference statistics for describing continuous features as
            "High"/"Low".
        waterfall_path: Destination path for the waterfall plot.
        top_n: Number of top contributing factors to report.

    Returns:
        A dictionary with ``prediction``, ``confidence_pct``,
        ``top_factors``, ``recommendation``, ``narrative``, and
        ``waterfall_plot_path``.

    Raises:
        ValueError: If `transaction` does not contain exactly one row.
    """
    result = explainer.explain_transaction(transaction)
    prediction = result["prediction"]
    fraud_probability = result["fraud_probability"] if result["fraud_probability"] is not None else 0.0
    confidence_pct = (fraud_probability if prediction == 1 else 1 - fraud_probability) * 100

    raw_feature_names = explainer.artifacts.feature_schema["raw_feature_names"]
    reference_cols = [c for c in reference_sample.columns if c in raw_feature_names]
    reference_encoded = explainer.encode(reference_sample[reference_cols])
    reference_medians = reference_encoded.median()

    top_factors = _top_contributing_factors(
        result["shap_explanation"], reference_medians, top_n=top_n
    )

    prediction_label = "Fraud" if prediction == 1 else "Not Fraud"
    recommendation = (
        "Flag transaction for manual review."
        if prediction == 1
        else "No action required; continue routine monitoring."
    )

    saved_waterfall_path = visualization.plot_shap_waterfall(
        result["shap_explanation"], waterfall_path
    )

    narrative = _build_narrative(prediction_label, confidence_pct, top_factors, recommendation)

    logger.info(
        "Generated local explanation: prediction=%s confidence=%.1f%% top_factors=%s",
        prediction_label,
        confidence_pct,
        top_factors,
    )

    return {
        "prediction": prediction_label,
        "confidence_pct": confidence_pct,
        "top_factors": top_factors,
        "recommendation": recommendation,
        "narrative": narrative,
        "waterfall_plot_path": str(saved_waterfall_path),
    }
