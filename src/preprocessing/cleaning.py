"""Data validation and quality checks.

Functions here only *inspect* and *report* on data quality (schema,
missing values, duplicates) — they do not engineer new features or encode
existing ones. See :mod:`src.preprocessing.feature_engineering` and
:mod:`src.preprocessing.encoding` for those steps.
"""

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

#: Columns expected to be present in the raw PaySim dataset.
REQUIRED_COLUMNS: list[str] = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]


def validate_required_columns(
    df: pd.DataFrame, required_columns: list[str] = REQUIRED_COLUMNS
) -> None:
    """Verify that all required columns are present in the dataset.

    Args:
        df: The DataFrame to validate.
        required_columns: Column names that must be present. Defaults to
            :data:`REQUIRED_COLUMNS`.

    Raises:
        ValueError: If one or more required columns are missing.
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    logger.info("All %d required columns are present.", len(required_columns))


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Compute missing-value counts and percentages per column.

    Args:
        df: The DataFrame to inspect.

    Returns:
        A DataFrame indexed by column name with ``missing_count`` and
        ``missing_pct`` columns, sorted by ``missing_count`` descending.
    """
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df)) * 100 if len(df) else missing_count.astype(float)

    summary = pd.DataFrame(
        {"missing_count": missing_count, "missing_pct": missing_pct}
    ).sort_values("missing_count", ascending=False)

    logger.info("Total missing cells: %d.", int(missing_count.sum()))
    return summary


def check_duplicates(df: pd.DataFrame) -> int:
    """Count fully duplicated rows in the dataset.

    Args:
        df: The DataFrame to inspect.

    Returns:
        The number of duplicated rows (all columns identical).
    """
    n_duplicates = int(df.duplicated().sum())
    logger.info("Found %d duplicated rows.", n_duplicates)
    return n_duplicates


def generate_cleaning_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Run all cleaning checks and assemble a single summary report.

    Args:
        df: The DataFrame to inspect.

    Returns:
        A dictionary containing dataset shape, missing-value summary, and
        duplicate row count. Intended for logging or persisting as a
        cleaning report.
    """
    missing_summary = check_missing_values(df)
    n_duplicates = check_duplicates(df)

    summary: dict[str, Any] = {
        "n_rows": df.shape[0],
        "n_columns": df.shape[1],
        "n_missing_cells": int(missing_summary["missing_count"].sum()),
        "columns_with_missing": missing_summary[
            missing_summary["missing_count"] > 0
        ].index.tolist(),
        "n_duplicate_rows": n_duplicates,
    }

    logger.info("Cleaning summary: %s", summary)
    return summary
