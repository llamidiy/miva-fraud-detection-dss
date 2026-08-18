"""Explainability service.

Two kinds of data here, handled differently:

- Global feature importance / SHAP summary — pre-computed by Sprint 5
  and frozen (`reports/feature_importance.csv`,
  `reports/shap/shap_values.joblib`, `reports/figures/*.png`). This
  service only *reads* those files; it never recomputes SHAP over the
  representative sample.
- Local (single-transaction) explanation / waterfall plot — cannot be
  pre-computed since the transaction is arbitrary user input, so these
  call the frozen `FraudExplainer`/`explain_single_transaction` live,
  exactly as Sprint 5 built them.
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

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.config import FEATURE_IMPORTANCE_PATH, FIGURES_DIR, REPORTS_DIR, SHAP_SAMPLE_PATH, SHAP_VALUES_PATH
from src.explainability.explainer import FraudExplainer
from src.explainability.local_explanations import explain_single_transaction
from src.validation.recommendation_engine import generate_recommendation
from src.validation.validator import build_transaction_dataframe

logger = logging.getLogger(__name__)

#: Destination for ad-hoc, single-transaction waterfall plots generated live
#: by the UI. Kept outside `reports/figures/` (Sprint 5's frozen output) to
#: avoid any confusion with the pipeline's own generated artifacts.
_UI_WATERFALL_PATH = REPORTS_DIR / "ui" / "single_transaction_waterfall.png"

#: Features with a pre-generated SHAP dependence plot (see Sprint 5).
_DEPENDENCE_PLOT_FEATURES = ("amount", "balanceDeltaOrig")


@st.cache_resource(show_spinner="Loading champion model...")
def get_champion_explainer() -> FraudExplainer:
    """Load and cache the champion model's SHAP explainer for this session.

    Defined here (rather than in `state.py`) so that `state.py` — which
    this module's siblings (`metrics_service`, `validation_service`,
    `report_service`) are imported by — never needs to be imported back
    by this module, avoiding a circular import.

    Returns:
        A `FraudExplainer` wrapping the frozen XGBoost deployment
        artifacts (model, encoder, metadata, feature schema).
    """
    logger.info("Loading champion model explainer...")
    start = time.perf_counter()
    explainer = FraudExplainer()
    logger.info("Champion model explainer loaded in %.3fs.", time.perf_counter() - start)
    return explainer


@dataclass
class FeatureImportanceRow:
    """One row of the global feature importance table."""

    feature: str
    importance: float


@st.cache_data(show_spinner="Loading feature importance...")
def get_feature_importance() -> list[FeatureImportanceRow]:
    """Load the champion model's pre-computed native feature importance.

    Reads Sprint 5's frozen `reports/feature_importance.csv`; does not
    recompute anything.

    Returns:
        Feature importance rows, ranked descending. Empty on failure.
    """
    try:
        if not FEATURE_IMPORTANCE_PATH.exists():
            raise FileNotFoundError(f"Feature importance not found at {FEATURE_IMPORTANCE_PATH}")
        df = pd.read_csv(FEATURE_IMPORTANCE_PATH)
        return [
            FeatureImportanceRow(feature=str(row.feature), importance=float(row.importance))
            for row in df.itertuples()
        ]
    except Exception:
        logger.exception("Failed to load feature importance.")
        return []


@dataclass
class ShapSummary:
    """Pre-generated global SHAP artifacts, ready for display.

    Attributes:
        summary_plot_path: Path to the SHAP summary plot, if generated.
        beeswarm_plot_path: Path to the SHAP beeswarm plot, if generated.
        bar_plot_path: Path to the mean(|SHAP|) bar plot, if generated.
        dependence_plot_paths: Feature name -> dependence plot path.
        top_features: Top features by mean |SHAP| value across the
            Sprint 5 representative sample.
        error: Set if the summary could not be loaded.
    """

    summary_plot_path: Optional[str]
    beeswarm_plot_path: Optional[str]
    bar_plot_path: Optional[str]
    dependence_plot_paths: dict[str, str] = field(default_factory=dict)
    top_features: list[FeatureImportanceRow] = field(default_factory=list)
    error: Optional[str] = None


def _existing_path_or_none(path: Path) -> Optional[str]:
    return str(path) if path.exists() else None


@st.cache_data(show_spinner="Loading SHAP summary...")
def get_shap_summary(top_n: int = 10) -> ShapSummary:
    """Load Sprint 5's pre-generated global SHAP figures and top features.

    Reads the cached `reports/shap/shap_values.joblib` and the figure
    files under `reports/figures/`; does not recompute SHAP.

    Args:
        top_n: Number of top mean-|SHAP| features to include.

    Returns:
        A `ShapSummary`. On failure, `error` is set and paths are ``None``.
    """
    try:
        if not SHAP_VALUES_PATH.exists():
            raise FileNotFoundError(f"Cached SHAP values not found at {SHAP_VALUES_PATH}")

        cached = joblib.load(SHAP_VALUES_PATH)
        explanation = cached["explanation"]
        mean_abs = np.abs(explanation.values).mean(axis=0)
        ranked = sorted(zip(explanation.feature_names, mean_abs), key=lambda item: item[1], reverse=True)

        return ShapSummary(
            summary_plot_path=_existing_path_or_none(FIGURES_DIR / "shap_summary.png"),
            beeswarm_plot_path=_existing_path_or_none(FIGURES_DIR / "shap_beeswarm.png"),
            bar_plot_path=_existing_path_or_none(FIGURES_DIR / "shap_bar.png"),
            dependence_plot_paths={
                feature: str(FIGURES_DIR / f"shap_dependence_{feature}.png")
                for feature in _DEPENDENCE_PLOT_FEATURES
                if (FIGURES_DIR / f"shap_dependence_{feature}.png").exists()
            },
            top_features=[
                FeatureImportanceRow(feature=name, importance=float(value)) for name, value in ranked[:top_n]
            ],
        )
    except Exception as exc:
        logger.exception("Failed to load SHAP summary.")
        return ShapSummary(
            summary_plot_path=None, beeswarm_plot_path=None, bar_plot_path=None, error=str(exc)
        )


@st.cache_data(show_spinner="Loading reference sample...")
def _load_reference_sample() -> pd.DataFrame:
    """Load Sprint 5's frozen representative sample, used for local explanation context."""
    return pd.read_csv(SHAP_SAMPLE_PATH)


@dataclass
class LocalExplanation:
    """A full SHAP-based local explanation for one ad-hoc transaction.

    Attributes:
        prediction: ``"Fraud"`` or ``"Not Fraud"``.
        confidence: Confidence in `prediction`, as a percentage.
        recommendation_tier: The recommendation engine's tier for this result.
        recommendation_actions: The recommended actions for that tier.
        top_factors: Ranked, human-readable SHAP contributing factors.
        narrative: The dashboard-ready natural-language explanation.
        waterfall_path: Path to the saved waterfall plot.
        error: Set if explanation failed; other fields hold safe defaults.
    """

    prediction: str
    confidence: float
    recommendation_tier: str
    recommendation_actions: list[str]
    top_factors: list[str]
    narrative: str
    waterfall_path: str
    error: Optional[str] = None


def get_local_explanation(transaction: dict[str, Any]) -> LocalExplanation:
    """Compute a full SHAP-based local explanation for one ad-hoc transaction.

    Live computation (cannot be pre-generated, since the transaction is
    arbitrary user input): builds the engineered feature row via the
    frozen `build_transaction_dataframe`, then calls the frozen
    `explain_single_transaction` and `generate_recommendation` exactly
    as Sprint 5/5.5 do. No prediction or SHAP logic is reimplemented here.

    Args:
        transaction: Raw transaction fields (see
            `src.validation.scenario_generator.RAW_TRANSACTION_FIELDS`).

    Returns:
        A `LocalExplanation`. On failure, `error` is set and other
        fields hold safe defaults.
    """
    start = time.perf_counter()
    try:
        explainer = get_champion_explainer()
        transaction_df = build_transaction_dataframe(transaction)
        reference_sample = _load_reference_sample()

        result = explain_single_transaction(
            explainer,
            transaction_df,
            reference_sample=reference_sample,
            waterfall_path=_UI_WATERFALL_PATH,
        )
        recommendation = generate_recommendation(result["prediction"], result["confidence_pct"])

        elapsed = time.perf_counter() - start
        logger.info(
            "Local explanation computed in %.3fs: prediction=%s confidence=%.1f%%",
            elapsed,
            result["prediction"],
            result["confidence_pct"],
        )
        return LocalExplanation(
            prediction=result["prediction"],
            confidence=round(result["confidence_pct"], 2),
            recommendation_tier=recommendation.tier,
            recommendation_actions=recommendation.actions,
            top_factors=result["top_factors"],
            narrative=result["narrative"],
            waterfall_path=result["waterfall_plot_path"],
        )
    except Exception as exc:
        logger.exception("Failed to compute local explanation after %.3fs.", time.perf_counter() - start)
        return LocalExplanation(
            prediction="N/A",
            confidence=0.0,
            recommendation_tier="N/A",
            recommendation_actions=[],
            top_factors=[],
            narrative="",
            waterfall_path="",
            error=str(exc),
        )


def get_waterfall(transaction: dict[str, Any]) -> Optional[str]:
    """Return the waterfall plot path for one ad-hoc transaction.

    Convenience wrapper around `get_local_explanation` for callers that
    only need the plot path. Note this recomputes the full local
    explanation; callers that also need the prediction/factors/narrative
    should call `get_local_explanation` directly instead of calling both.

    Args:
        transaction: Raw transaction fields, as for `get_local_explanation`.

    Returns:
        The waterfall plot path, or ``None`` if explanation failed.
    """
    explanation = get_local_explanation(transaction)
    return explanation.waterfall_path if not explanation.error else None
