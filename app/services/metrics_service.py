"""Model metrics service.

Loads the champion model's evaluation metrics from Sprint 4/4.5's
existing, frozen reports (`reports/model_results.csv` and
`models/<name>/metadata.json`) — no model is retrained or re-evaluated
here.
"""

import json
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
import streamlit as st

from src.config import CHAMPION_MODEL_NAME, MODEL_DIR, MODEL_RESULTS_PATH

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """UI-friendly summary of the champion model's evaluation metrics.

    Attributes:
        model_name: Registered model name (e.g. ``"xgboost"``).
        algorithm: The underlying estimator class name.
        accuracy: Test-set accuracy, or ``None`` if unavailable.
        precision: Test-set precision, or ``None`` if unavailable.
        recall: Test-set recall, or ``None`` if unavailable.
        f1_score: Test-set F1-score, or ``None`` if unavailable.
        roc_auc: Test-set ROC-AUC, or ``None`` if unavailable.
        training_time_seconds: Training wall-clock time, or ``None``.
        comparison: All trained models' metric rows, for a comparison table.
        error: Set if metrics could not be loaded; other fields hold
            safe defaults in that case.
    """

    model_name: str
    algorithm: str
    accuracy: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    f1_score: Optional[float]
    roc_auc: Optional[float]
    training_time_seconds: Optional[float]
    comparison: list[dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


def _load_algorithm_name(model_name: str) -> str:
    """Read the algorithm name from a model's persisted metadata.json, if available."""
    metadata_path = MODEL_DIR / model_name / "metadata.json"
    if not metadata_path.exists():
        return model_name
    try:
        with metadata_path.open(encoding="utf-8") as f:
            return json.load(f).get("algorithm", model_name)
    except (OSError, json.JSONDecodeError):
        logger.warning("Could not read algorithm name from %s.", metadata_path)
        return model_name


@st.cache_data(show_spinner="Loading model metrics...")
def get_model_metrics(champion_model_name: str = CHAMPION_MODEL_NAME) -> ModelMetrics:
    """Load the champion model's metrics summary and the full model comparison table.

    Args:
        champion_model_name: Registered name of the champion model.

    Returns:
        A `ModelMetrics` summary. On failure, `error` is set and numeric
        fields default to ``None``.
    """
    start = time.perf_counter()
    try:
        if not MODEL_RESULTS_PATH.exists():
            raise FileNotFoundError(f"Model results not found at {MODEL_RESULTS_PATH}")

        df = pd.read_csv(MODEL_RESULTS_PATH)
        champion_rows = df[df["model_name"] == champion_model_name]
        if champion_rows.empty:
            raise ValueError(
                f"Champion model '{champion_model_name}' not found in {MODEL_RESULTS_PATH}"
            )
        row = champion_rows.iloc[0]

        metrics = ModelMetrics(
            model_name=champion_model_name,
            algorithm=_load_algorithm_name(champion_model_name),
            accuracy=float(row["accuracy"]),
            precision=float(row["precision"]),
            recall=float(row["recall"]),
            f1_score=float(row["f1_score"]),
            roc_auc=float(row["roc_auc"]) if pd.notna(row["roc_auc"]) else None,
            training_time_seconds=float(row["training_time_seconds"]),
            comparison=df[
                ["model_name", "accuracy", "precision", "recall", "f1_score", "roc_auc", "training_time_seconds"]
            ].to_dict("records"),
        )
        logger.info("Loaded model metrics in %.3fs.", time.perf_counter() - start)
        return metrics

    except Exception as exc:
        logger.exception("Failed to load model metrics.")
        return ModelMetrics(
            model_name=champion_model_name,
            algorithm="Unavailable",
            accuracy=None,
            precision=None,
            recall=None,
            f1_score=None,
            roc_auc=None,
            training_time_seconds=None,
            comparison=[],
            error=str(exc),
        )
