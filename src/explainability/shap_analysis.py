"""Global SHAP analysis over the representative sample.

Computes SHAP values once for the explainability sample (caching the
result to avoid expensive recomputation on repeated runs) and generates
the full set of global SHAP figures via `visualization.py`.
"""

import hashlib
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import FIGURES_DIR, SHAP_VALUES_PATH, TARGET_COLUMN
from src.explainability import visualization
from src.explainability.explainer import FraudExplainer

logger = logging.getLogger(__name__)

#: Features plotted individually via SHAP dependence plots.
DEPENDENCE_PLOT_FEATURES: list[str] = ["amount", "balanceDeltaOrig"]


def _sample_fingerprint(sample: pd.DataFrame) -> str:
    """Compute a content hash of the sample for cache validation.

    Hashes the sample's CSV serialization rather than the in-memory
    DataFrame directly, so the fingerprint is stable whether `sample`
    came fresh from a split or was reloaded from the cached sample CSV
    (which can otherwise produce different dtypes for logically
    identical values and defeat cache matching).
    """
    return hashlib.sha256(sample.to_csv(index=False).encode("utf-8")).hexdigest()


def compute_or_load_shap_values(
    explainer: FraudExplainer,
    sample: pd.DataFrame,
    cache_path: Path = SHAP_VALUES_PATH,
    use_cache: bool = True,
) -> Any:
    """Compute SHAP values for the sample, reusing a valid cached result if available.

    The cache is keyed on the sample's row count, a content fingerprint,
    and the model's encoded feature names — if any of these differ from
    what's cached, the cache is treated as stale and SHAP values are
    recomputed.

    Args:
        explainer: An initialized `FraudExplainer`.
        sample: The representative sample to explain (raw, pre-encoding
            columns, may include the target column).
        cache_path: Location of the cached SHAP values file.
        use_cache: If ``False``, always recompute and overwrite the cache.

    Returns:
        A `shap.Explanation` for the sample.
    """
    X_raw = sample.drop(columns=[TARGET_COLUMN]) if TARGET_COLUMN in sample.columns else sample
    fingerprint = _sample_fingerprint(X_raw)
    feature_names = explainer.artifacts.feature_schema["encoded_feature_order"]

    if use_cache and cache_path.exists():
        try:
            cached = joblib.load(cache_path)
        except (OSError, EOFError, ValueError):
            cached = None

        if (
            cached is not None
            and cached.get("fingerprint") == fingerprint
            and cached.get("feature_names") == feature_names
        ):
            logger.info(
                "Reusing cached SHAP values from %s (%d rows); skipping recomputation.",
                cache_path,
                len(X_raw),
            )
            return cached["explanation"]
        logger.info("Cached SHAP values at %s are stale or missing; recomputing.", cache_path)

    explanation = explainer.get_shap_values(X_raw)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"explanation": explanation, "fingerprint": fingerprint, "feature_names": feature_names},
        cache_path,
    )
    logger.info("Cached SHAP values to %s.", cache_path)
    return explanation


def run_shap_analysis(
    explainer: FraudExplainer, sample: pd.DataFrame, use_cache: bool = True
) -> dict[str, Any]:
    """Compute SHAP values for the sample and generate all global SHAP figures.

    Args:
        explainer: An initialized `FraudExplainer`.
        sample: The representative explainability sample.
        use_cache: Whether to reuse a valid cached SHAP computation.

    Returns:
        A dictionary with the ``explanation`` object and the paths of
        every figure generated (``shap_summary``, ``shap_beeswarm``,
        ``shap_bar``, and one ``shap_dependence_<feature>`` entry per
        feature in `DEPENDENCE_PLOT_FEATURES`).
    """
    X_raw = sample.drop(columns=[TARGET_COLUMN]) if TARGET_COLUMN in sample.columns else sample

    explanation = compute_or_load_shap_values(explainer, sample, use_cache=use_cache)
    X_encoded = explainer.encode(X_raw)

    figures: dict[str, Path] = {
        "shap_summary": visualization.plot_shap_summary(
            explanation.values, X_encoded, FIGURES_DIR / "shap_summary.png"
        ),
        "shap_beeswarm": visualization.plot_shap_beeswarm(
            explanation, FIGURES_DIR / "shap_beeswarm.png"
        ),
        "shap_bar": visualization.plot_shap_bar(explanation, FIGURES_DIR / "shap_bar.png"),
    }

    for feature in DEPENDENCE_PLOT_FEATURES:
        key = f"shap_dependence_{feature}"
        figures[key] = visualization.plot_shap_dependence(
            feature, explanation.values, X_encoded, FIGURES_DIR / f"{key}.png"
        )

    logger.info("Generated %d SHAP figures for the representative sample.", len(figures))
    return {"explanation": explanation, "figures": figures}
