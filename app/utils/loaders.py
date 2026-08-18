"""Placeholder data-loading functions for the Phase 1 application shell.

Every function here returns a static mock value only — none of them
read the filesystem, call the trained model, or import anything from
`src.*`. Each docstring states exactly what Phase 2 will load instead,
and every function keeps the same name and return shape it will have
once wired to the real backend, so pages built against these loaders
should need minimal changes later.

Each loader is wrapped in a broad `try/except` and falls back to a safe
empty value on failure, so a page can always render something (an empty
state) instead of crashing — the same defensive shape Phase 2's real
file/model-loading code will need.
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def load_dashboard_metrics() -> dict[str, str]:
    """Return placeholder headline metrics for the Dashboard page.

    Phase 2 will compute these from `reports/model_results.csv` and
    `reports/validation/scenario_results.csv`.

    Returns:
        A dict of mock metric values, or an empty dict on failure.
    """
    try:
        return {
            "total_transactions_scored": "Loading...",
            "fraud_flagged": "Loading...",
            "champion_model_accuracy": "Loading...",
            "avg_prediction_confidence": "Loading...",
        }
    except Exception:
        logger.exception("Failed to load mock dashboard metrics.")
        return {}


def load_champion_model_metadata() -> Optional[dict[str, Any]]:
    """Return placeholder champion-model metadata.

    Phase 2 will load `models/xgboost/metadata.json`.

    Returns:
        A mock metadata dict, or ``None`` if unavailable.
    """
    try:
        return {
            "model_name": "Loading...",
            "algorithm": "Loading...",
            "training_timestamp": "Loading...",
            "target_column": "Loading...",
        }
    except Exception:
        logger.exception("Failed to load mock champion model metadata.")
        return None


def load_model_comparison() -> list[dict[str, Any]]:
    """Return a placeholder (empty) model comparison table.

    Phase 2 will load `reports/model_results.csv`.

    Returns:
        An empty list of row dicts.
    """
    try:
        return []
    except Exception:
        logger.exception("Failed to load mock model comparison table.")
        return []


def load_scenario_results() -> list[dict[str, Any]]:
    """Return a placeholder (empty) validation scenario results table.

    Phase 2 will load `reports/validation/scenario_results.csv`.

    Returns:
        An empty list of row dicts.
    """
    try:
        return []
    except Exception:
        logger.exception("Failed to load mock scenario results.")
        return []


def load_feature_importance() -> list[dict[str, Any]]:
    """Return a placeholder (empty) global feature importance table.

    Phase 2 will load `reports/feature_importance.csv`.

    Returns:
        An empty list of row dicts.
    """
    try:
        return []
    except Exception:
        logger.exception("Failed to load mock feature importance table.")
        return []


def load_batch_result_columns() -> list[str]:
    """Return the expected column headers for a batch assessment result table.

    Phase 2 will populate this table by scoring an uploaded CSV through
    `src.models.predictor` and `src.explainability`.

    Returns:
        A list of expected column names, or an empty list on failure.
    """
    try:
        return [
            "transaction_id",
            "prediction",
            "confidence",
            "recommendation",
            "top_factor",
        ]
    except Exception:
        logger.exception("Failed to load batch result column definitions.")
        return []
