"""Representative sampling for explainability computations.

SHAP is expensive to compute; running it against the full 6.36M-row
engineered dataset is impractical. This module draws a small, stratified
sample that preserves the fraud/non-fraud ratio of the full dataset and
caches it to disk, so every explainability module works against the
exact same representative sample rather than each drawing its own.
"""

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import EXPLAINABILITY_SAMPLE_SIZE, RANDOM_STATE, SHAP_SAMPLE_PATH, TARGET_COLUMN
from src.preprocessing.loader import load_engineered_dataset

logger = logging.getLogger(__name__)


def create_explainability_sample(
    sample_size: int = EXPLAINABILITY_SAMPLE_SIZE,
    target_column: str = TARGET_COLUMN,
    random_state: int = RANDOM_STATE,
    path: Path = SHAP_SAMPLE_PATH,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Draw (or reuse) a stratified representative sample for explainability.

    Args:
        sample_size: Number of rows to sample. Defaults to
            `EXPLAINABILITY_SAMPLE_SIZE` (10,000).
        target_column: Name of the binary target column to stratify on,
            so the fraud/non-fraud ratio of the full dataset is preserved
            in the sample.
        random_state: Seed for reproducibility.
        path: Destination CSV path. Defaults to the configured
            explainability sample path (`data/explainability/shap_sample.csv`).
        use_cache: If ``True`` and a cached sample of exactly
            `sample_size` rows already exists at `path`, reuse it instead
            of re-sampling from the full dataset.

    Returns:
        The sampled DataFrame, with the target column's class ratio
        matching the full dataset.

    Raises:
        ValueError: If `sample_size` is not a positive integer.
    """
    if sample_size <= 0:
        raise ValueError(f"sample_size must be positive, got {sample_size}.")

    if use_cache and path.exists():
        cached = pd.read_csv(path)
        if len(cached) == sample_size:
            logger.info(
                "Reusing cached explainability sample at %s (%d rows).", path, len(cached)
            )
            return cached
        logger.info(
            "Cached sample at %s has %d rows, not the requested %d; regenerating.",
            path,
            len(cached),
            sample_size,
        )

    df = load_engineered_dataset()

    if sample_size >= len(df):
        logger.warning(
            "Requested sample_size=%d >= dataset size=%d; using the full dataset.",
            sample_size,
            len(df),
        )
        sample = df.copy()
    else:
        sample, _ = train_test_split(
            df,
            train_size=sample_size,
            stratify=df[target_column],
            random_state=random_state,
        )

    sample = sample.reset_index(drop=True)

    fraud_rate_full = df[target_column].mean()
    fraud_rate_sample = sample[target_column].mean()
    logger.info(
        "Created explainability sample: %d rows (fraud rate %.4f%%, full dataset %.4f%%).",
        len(sample),
        fraud_rate_sample * 100,
        fraud_rate_full * 100,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(path, index=False)
    logger.info("Saved explainability sample to %s.", path)

    return sample
