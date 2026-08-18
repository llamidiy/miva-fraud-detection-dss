"""Dataset loading and persistence utilities.

Thin, well-documented wrappers around pandas I/O that centralize error
handling and default file locations for the fraud detection dataset.
"""

import logging
from pathlib import Path

import pandas as pd

from src.config import ENGINEERED_DATASET_PATH, RAW_DATASET_PATH

logger = logging.getLogger(__name__)


def load_raw_dataset(path: Path = RAW_DATASET_PATH) -> pd.DataFrame:
    """Load the raw PaySim transaction dataset from disk.

    Args:
        path: Location of the raw CSV file. Defaults to the configured
            raw dataset path.

    Returns:
        The raw dataset as a DataFrame, unmodified.

    Raises:
        FileNotFoundError: If no file exists at ``path``.
        ValueError: If the file exists but is empty or cannot be parsed
            as CSV.
    """
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found at: {path}")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Raw dataset at {path} is empty.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Raw dataset at {path} could not be parsed as CSV.") from exc

    logger.info("Loaded raw dataset from %s (%d rows, %d columns).", path, *df.shape)
    return df


def load_engineered_dataset(path: Path = ENGINEERED_DATASET_PATH) -> pd.DataFrame:
    """Load a previously saved, feature-engineered dataset from disk.

    The engineered dataset contains cleaned data plus engineered features,
    but categorical columns are not yet encoded — encoding happens
    separately during model training/inference (see
    :mod:`src.preprocessing.encoding`).

    Args:
        path: Location of the engineered CSV file. Defaults to the
            configured engineered dataset path.

    Returns:
        The engineered dataset as a DataFrame.

    Raises:
        FileNotFoundError: If no file exists at ``path``.
        ValueError: If the file exists but is empty or cannot be parsed
            as CSV.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Engineered dataset not found at: {path}. Run the preprocessing "
            "pipeline first."
        )

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Engineered dataset at {path} is empty.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Engineered dataset at {path} could not be parsed as CSV.") from exc

    logger.info("Loaded engineered dataset from %s (%d rows, %d columns).", path, *df.shape)
    return df


def save_engineered_dataset(df: pd.DataFrame, path: Path = ENGINEERED_DATASET_PATH) -> Path:
    """Persist a feature-engineered dataset to disk as CSV.

    Creates the destination directory if it does not already exist.

    Args:
        df: The engineered DataFrame to save.
        path: Destination file path. Defaults to the configured engineered
            dataset path.

    Returns:
        The path the dataset was written to.

    Raises:
        ValueError: If ``df`` is empty.
        OSError: If the file cannot be written (e.g. permissions issue).
    """
    if df.empty:
        raise ValueError("Refusing to save an empty DataFrame.")

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    except OSError as exc:
        raise OSError(f"Failed to write engineered dataset to {path}: {exc}") from exc

    logger.info("Saved engineered dataset to %s (%d rows, %d columns).", path, *df.shape)
    return path
