"""Centralized plotting utilities for the explainability layer.

Every figure the explainability layer produces is generated here — no
other module in `src.explainability` should call matplotlib or SHAP's
plotting functions directly. Keeping all rendering logic in one place
makes styling consistent and easy to change project-wide.
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering; no display required

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


def _save_current_figure(path: Path) -> Path:
    """Save and close the current matplotlib figure.

    Args:
        path: Destination file path. Its parent directory is created
            automatically if it does not already exist.

    Returns:
        The path the figure was written to.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    logger.info("Saved figure to %s.", path)
    return path


def plot_feature_importance(importance_df: pd.DataFrame, path: Path, top_n: int = 20) -> Path:
    """Plot a horizontal bar chart of the top-N most important features.

    Args:
        importance_df: A DataFrame with ``feature`` and ``importance``
            columns, sorted descending by importance.
        path: Destination PNG path.
        top_n: Number of top features to display.

    Returns:
        The path the figure was written to.
    """
    top = importance_df.head(top_n).iloc[::-1]
    plt.figure(figsize=(9, max(4, 0.35 * len(top))))
    plt.barh(top["feature"], top["importance"], color="steelblue")
    plt.xlabel("Importance (gain)")
    plt.title("XGBoost Global Feature Importance")
    return _save_current_figure(path)


def plot_shap_summary(shap_values: np.ndarray, X: pd.DataFrame, path: Path) -> Path:
    """Plot the classic SHAP summary (dot) plot.

    Args:
        shap_values: SHAP values as a 2D array `(n_samples, n_features)`.
        X: The corresponding encoded feature matrix.
        path: Destination PNG path.

    Returns:
        The path the figure was written to.
    """
    plt.figure(figsize=(9, 7))
    shap.summary_plot(shap_values, X, show=False)
    return _save_current_figure(path)


def plot_shap_beeswarm(explanation: shap.Explanation, path: Path) -> Path:
    """Plot a SHAP beeswarm plot from an `Explanation` object.

    Args:
        explanation: A multi-row `shap.Explanation`.
        path: Destination PNG path.

    Returns:
        The path the figure was written to.
    """
    plt.figure(figsize=(9, 7))
    shap.plots.beeswarm(explanation, show=False)
    return _save_current_figure(path)


def plot_shap_bar(explanation: shap.Explanation, path: Path) -> Path:
    """Plot a mean(|SHAP value|) bar chart from an `Explanation` object.

    Args:
        explanation: A multi-row `shap.Explanation`.
        path: Destination PNG path.

    Returns:
        The path the figure was written to.
    """
    plt.figure(figsize=(9, 7))
    shap.plots.bar(explanation, show=False)
    return _save_current_figure(path)


def plot_shap_dependence(
    feature_name: str, shap_values: np.ndarray, X: pd.DataFrame, path: Path
) -> Path:
    """Plot a SHAP dependence plot for a single feature.

    Args:
        feature_name: Name of the feature to plot (must be a column of
            `X`).
        shap_values: SHAP values as a 2D array `(n_samples, n_features)`,
            aligned with `X`'s columns.
        X: The corresponding encoded feature matrix.
        path: Destination PNG path.

    Returns:
        The path the figure was written to.
    """
    plt.figure(figsize=(8, 6))
    shap.dependence_plot(feature_name, shap_values, X, show=False)
    return _save_current_figure(path)


def plot_shap_waterfall(explanation_single: shap.Explanation, path: Path) -> Path:
    """Plot a SHAP waterfall plot for a single transaction.

    Args:
        explanation_single: A single-row `shap.Explanation` (e.g.
            `explanation[0]`).
        path: Destination PNG path.

    Returns:
        The path the figure was written to.
    """
    plt.figure(figsize=(9, 6))
    shap.plots.waterfall(explanation_single, show=False)
    return _save_current_figure(path)
