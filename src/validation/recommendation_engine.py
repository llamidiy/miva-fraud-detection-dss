"""Deterministic recommendation rules for fraud predictions.

Recommendations are derived solely from two inputs — the predicted
class and the model's confidence in that prediction — via fixed
thresholds. The rules are intentionally simple and fully transparent so
they can be audited independently of the model itself:

    Fraud, confidence >= HIGH_CONFIDENCE_THRESHOLD   -> flag/suspend/escalate
    Fraud, confidence <  HIGH_CONFIDENCE_THRESHOLD   -> verify/monitor
    Not Fraud (any confidence)                        -> approve/monitor

No machine learning happens here; this module only encodes business
policy on top of the champion model's output.
"""

import logging
from dataclasses import dataclass

from src.config import HIGH_CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)

#: Actions for a fraud prediction at or above `HIGH_CONFIDENCE_THRESHOLD`.
FRAUD_HIGH_CONFIDENCE_ACTIONS: list[str] = [
    "Flag immediately",
    "Suspend pending investigation",
    "Escalate to fraud analyst",
]

#: Actions for a fraud prediction below `HIGH_CONFIDENCE_THRESHOLD`.
FRAUD_MEDIUM_CONFIDENCE_ACTIONS: list[str] = [
    "Request additional verification",
    "Monitor subsequent activity",
]

#: Actions for a legitimate ("Not Fraud") prediction, at any confidence.
LEGITIMATE_ACTIONS: list[str] = [
    "Approve transaction",
    "Continue routine monitoring",
]

#: Recommendation tier identifiers, used to compare expected vs. actual outcomes.
TIER_FRAUD_HIGH_CONFIDENCE = "fraud_high_confidence"
TIER_FRAUD_MEDIUM_CONFIDENCE = "fraud_medium_confidence"
TIER_LEGITIMATE = "legitimate"


@dataclass
class Recommendation:
    """A deterministic recommendation for one prediction.

    Attributes:
        tier: One of `TIER_FRAUD_HIGH_CONFIDENCE`,
            `TIER_FRAUD_MEDIUM_CONFIDENCE`, or `TIER_LEGITIMATE`.
        actions: Ordered list of recommended actions for this tier.
        rationale: A short, human-readable explanation of why this tier
            was chosen (states the exact rule that fired).
    """

    tier: str
    actions: list[str]
    rationale: str


def generate_recommendation(prediction_label: str, confidence_pct: float) -> Recommendation:
    """Generate a deterministic recommendation from a prediction and its confidence.

    Args:
        prediction_label: ``"Fraud"`` or ``"Not Fraud"``.
        confidence_pct: Confidence in `prediction_label`, as a percentage
            (0-100).

    Returns:
        The resulting `Recommendation`.

    Raises:
        ValueError: If `prediction_label` is not ``"Fraud"`` or ``"Not Fraud"``.
    """
    if prediction_label not in {"Fraud", "Not Fraud"}:
        raise ValueError(f"Unknown prediction_label '{prediction_label}'.")

    if prediction_label == "Fraud":
        if confidence_pct >= HIGH_CONFIDENCE_THRESHOLD:
            recommendation = Recommendation(
                tier=TIER_FRAUD_HIGH_CONFIDENCE,
                actions=list(FRAUD_HIGH_CONFIDENCE_ACTIONS),
                rationale=(
                    f"Predicted Fraud with {confidence_pct:.1f}% confidence, at or above the "
                    f"{HIGH_CONFIDENCE_THRESHOLD:.0f}% high-confidence threshold."
                ),
            )
        else:
            recommendation = Recommendation(
                tier=TIER_FRAUD_MEDIUM_CONFIDENCE,
                actions=list(FRAUD_MEDIUM_CONFIDENCE_ACTIONS),
                rationale=(
                    f"Predicted Fraud with {confidence_pct:.1f}% confidence, below the "
                    f"{HIGH_CONFIDENCE_THRESHOLD:.0f}% high-confidence threshold."
                ),
            )
    else:
        recommendation = Recommendation(
            tier=TIER_LEGITIMATE,
            actions=list(LEGITIMATE_ACTIONS),
            rationale=f"Predicted Not Fraud with {confidence_pct:.1f}% confidence.",
        )

    logger.info(
        "Recommendation: prediction=%s confidence=%.1f%% -> tier=%s actions=%s",
        prediction_label,
        confidence_pct,
        recommendation.tier,
        recommendation.actions,
    )
    return recommendation
