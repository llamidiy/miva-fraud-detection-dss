"""Model evaluation and cross-model comparison reporting.

Wraps the pure metric functions in :mod:`src.utils.metrics` with timing
instrumentation (training/prediction time) and assembles per-model
results into a single comparison table saved to
``reports/model_results.csv``.
"""

import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import MODEL_RESULTS_PATH
from src.models.base_model import BaseModel
from src.utils.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


def evaluate_model(
    model: BaseModel, X_test: pd.DataFrame, y_test: pd.Series, training_time: float
) -> dict[str, Any]:
    """Evaluate a fitted model on held-out test data.

    Runs prediction (timed), computes the full metric suite via
    :func:`~src.utils.metrics.compute_all_metrics`, and combines it with
    the supplied training time into a single result record.

    Args:
        model: A fitted model wrapper.
        X_test: Held-out test feature matrix.
        y_test: Held-out test target vector.
        training_time: Seconds spent fitting the model, measured by the
            caller (:mod:`src.models.trainer`).

    Returns:
        A dictionary containing ``model_name``, ``training_time_seconds``,
        ``prediction_time_seconds``, and every key produced by
        :func:`~src.utils.metrics.compute_all_metrics`.
    """
    start = time.perf_counter()
    y_pred = model.predict(X_test)
    prediction_time = time.perf_counter() - start

    y_score = model.predict_proba(X_test)
    metrics = compute_all_metrics(y_test, y_pred, y_score)

    result: dict[str, Any] = {
        "model_name": model.name,
        "training_time_seconds": round(training_time, 4),
        "prediction_time_seconds": round(prediction_time, 4),
        **metrics,
    }

    logger.info(
        "Evaluation for '%s': accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f roc_auc=%s "
        "train_time=%.2fs pred_time=%.4fs",
        model.name,
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1_score"],
        f"{metrics['roc_auc']:.4f}" if metrics["roc_auc"] is not None else "N/A",
        training_time,
        prediction_time,
    )
    return result


def save_comparison_table(
    results: list[dict[str, Any]], path: Path = MODEL_RESULTS_PATH
) -> pd.DataFrame:
    """Assemble per-model evaluation results into a comparison table and save it.

    Args:
        results: One result dictionary per model, as returned by
            :func:`evaluate_model`.
        path: Destination CSV path. Defaults to the configured model
            results path.

    Returns:
        The comparison table as a DataFrame (one row per model).

    Raises:
        ValueError: If ``results`` is empty.
        OSError: If the file cannot be written.
    """
    if not results:
        raise ValueError("Cannot save an empty model comparison table.")

    comparison_df = pd.DataFrame(results)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(path, index=False)
    except OSError as exc:
        raise OSError(f"Failed to write model comparison table to {path}: {exc}") from exc

    logger.info("Saved model comparison table (%d models) to %s.", len(results), path)
    return comparison_df
