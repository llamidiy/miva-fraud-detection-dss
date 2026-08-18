"""Global feature importance from the champion model's native scores.

Complements the SHAP-based analysis in `shap_analysis.py` with the
champion XGBoost model's built-in, gain-based feature importance — fast
to compute and independent of the representative sample.
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import FEATURE_IMPORTANCE_PATH, FIGURES_DIR
from src.explainability import visualization
from src.explainability.explainer import FraudExplainer

logger = logging.getLogger(__name__)


def compute_feature_importance(explainer: FraudExplainer) -> pd.DataFrame:
    """Compute the champion model's native (gain-based) feature importance.

    Args:
        explainer: An initialized `FraudExplainer`.

    Returns:
        A DataFrame with ``feature`` and ``importance`` columns, sorted
        descending by importance.
    """
    feature_names = explainer.artifacts.feature_schema["encoded_feature_order"]
    importances = explainer.artifacts.model.model.feature_importances_

    df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    logger.info("Computed native feature importance for %d features.", len(df))
    return df


def save_feature_importance_csv(df: pd.DataFrame, path: Path = FEATURE_IMPORTANCE_PATH) -> Path:
    """Persist the feature importance table to disk as CSV.

    Args:
        df: The feature importance DataFrame.
        path: Destination CSV path.

    Returns:
        The path the table was written to.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved feature importance table to %s.", path)
    return path


def run_feature_importance(explainer: FraudExplainer) -> dict[str, Any]:
    """Compute, save, and plot the champion model's global feature importance.

    Args:
        explainer: An initialized `FraudExplainer`.

    Returns:
        A dictionary with the importance ``dataframe``, the saved
        ``csv_path``, and the saved ``png_path``.
    """
    df = compute_feature_importance(explainer)
    csv_path = save_feature_importance_csv(df)
    png_path = visualization.plot_feature_importance(df, FIGURES_DIR / "feature_importance.png")

    return {"dataframe": df, "csv_path": csv_path, "png_path": png_path}
