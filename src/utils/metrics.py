"""Reusable classification metric computations.

Pure, stateless functions that wrap scikit-learn's metrics so evaluation
logic is defined once and shared by every model in
:mod:`src.models.evaluator`. Timing metrics (training/prediction time) are
process-level concerns and are computed by the evaluator, not here.
"""

import logging
from typing import Any, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute overall classification accuracy."""
    return float(accuracy_score(y_true, y_pred))


def compute_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute precision for the positive (fraud) class.

    Returns 0.0 instead of raising when there are no predicted positives.
    """
    return float(precision_score(y_true, y_pred, zero_division=0))


def compute_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute recall for the positive (fraud) class.

    Returns 0.0 instead of raising when there are no actual positives.
    """
    return float(recall_score(y_true, y_pred, zero_division=0))


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the F1-score for the positive (fraud) class."""
    return float(f1_score(y_true, y_pred, zero_division=0))


def compute_roc_auc(y_true: np.ndarray, y_score: Optional[np.ndarray]) -> Optional[float]:
    """Compute ROC-AUC from continuous fraud scores, where applicable.

    Args:
        y_true: Ground-truth binary labels.
        y_score: Continuous fraud scores/probabilities, or ``None`` if the
            model does not produce one.

    Returns:
        The ROC-AUC score, or ``None`` if it cannot be computed (no score
        available, or only one class present in ``y_true``).
    """
    if y_score is None:
        return None
    if len(np.unique(y_true)) < 2:
        logger.warning("ROC-AUC undefined: only one class present in y_true.")
        return None
    return float(roc_auc_score(y_true, y_score))


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    """Compute the 2x2 confusion matrix as a nested list (JSON/CSV friendly)."""
    return confusion_matrix(y_true, y_pred).tolist()


def compute_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Compute the full per-class precision/recall/F1 report as text."""
    return classification_report(y_true, y_pred, zero_division=0)


def compute_all_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_score: Optional[np.ndarray] = None
) -> dict[str, Any]:
    """Compute the complete set of classification metrics in one call.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.
        y_score: Continuous fraud scores/probabilities used for ROC-AUC,
            or ``None`` if unavailable.

    Returns:
        A dictionary with ``accuracy``, ``precision``, ``recall``,
        ``f1_score``, ``roc_auc``, ``confusion_matrix``, and
        ``classification_report`` keys.
    """
    return {
        "accuracy": compute_accuracy(y_true, y_pred),
        "precision": compute_precision(y_true, y_pred),
        "recall": compute_recall(y_true, y_pred),
        "f1_score": compute_f1(y_true, y_pred),
        "roc_auc": compute_roc_auc(y_true, y_score),
        "confusion_matrix": compute_confusion_matrix(y_true, y_pred),
        "classification_report": compute_classification_report(y_true, y_pred),
    }
