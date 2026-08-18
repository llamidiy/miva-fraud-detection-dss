"""End-to-end preprocessing pipeline orchestration.

Chains together loading, validation, cleaning checks, and feature
engineering, then persists the resulting *engineered* dataset to
``data/processed``. Categorical encoding is intentionally excluded: it is
performed later, during model training/inference, using
:mod:`src.preprocessing.encoding`. This keeps the saved dataset reusable
across different encoding strategies rather than baking one in.

On success, the pipeline also writes a JSON metadata file and a Markdown
report to ``reports/`` documenting the run.

Intended to be run as a script (``python -m src.preprocessing.pipeline``)
or imported and called via :func:`run_pipeline`.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import LOG_PATH, METADATA_PATH, RAW_DATASET_FILENAME, REPORT_PATH
from src.preprocessing.cleaning import generate_cleaning_summary, validate_required_columns
from src.preprocessing.feature_engineering import engineer_features
from src.preprocessing.loader import load_raw_dataset, save_engineered_dataset

logger = logging.getLogger(__name__)


def _build_metadata(
    cleaning_summary: dict[str, Any],
    engineered_feature_names: list[str],
    engineered_df: pd.DataFrame,
    output_path: Path,
    timestamp: str,
) -> dict[str, Any]:
    """Assemble the preprocessing run metadata dictionary.

    Args:
        cleaning_summary: Summary produced by
            :func:`~src.preprocessing.cleaning.generate_cleaning_summary`
            on the raw dataset.
        engineered_feature_names: Names of the columns added during
            feature engineering.
        engineered_df: The final engineered DataFrame, used for its shape.
        output_path: Path the engineered dataset was written to.
        timestamp: ISO-8601 UTC timestamp of the pipeline run.

    Returns:
        A JSON-serializable metadata dictionary.
    """
    return {
        "raw_dataset_filename": RAW_DATASET_FILENAME,
        "execution_timestamp": timestamp,
        "total_rows": int(engineered_df.shape[0]),
        "total_columns": int(engineered_df.shape[1]),
        "engineered_feature_names": engineered_feature_names,
        "n_engineered_features": len(engineered_feature_names),
        "duplicate_rows": cleaning_summary["n_duplicate_rows"],
        "missing_values": cleaning_summary["n_missing_cells"],
        "output_dataset_path": str(output_path),
    }


def _save_metadata(metadata: dict[str, Any], path: Path = METADATA_PATH) -> Path:
    """Write the preprocessing metadata dictionary to disk as JSON.

    Args:
        metadata: The metadata dictionary to persist.
        path: Destination file path. Defaults to the configured metadata
            path.

    Returns:
        The path the metadata was written to.

    Raises:
        OSError: If the file cannot be written.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    except OSError as exc:
        raise OSError(f"Failed to write preprocessing metadata to {path}: {exc}") from exc

    logger.info("Saved preprocessing metadata to %s.", path)
    return path


def _build_report(metadata: dict[str, Any]) -> str:
    """Render the preprocessing metadata as a Markdown report.

    Args:
        metadata: The metadata dictionary produced by :func:`_build_metadata`.

    Returns:
        The report content as a Markdown string.
    """
    engineered_features_list = "\n".join(
        f"- `{name}`" for name in metadata["engineered_feature_names"]
    ) or "- (none)"

    return f"""# Preprocessing Report

## Dataset Summary

- **Raw dataset filename:** `{metadata['raw_dataset_filename']}`
- **Total rows:** {metadata['total_rows']:,}
- **Total columns:** {metadata['total_columns']}

## Cleaning Summary

- **Duplicate rows:** {metadata['duplicate_rows']:,}
- **Missing values:** {metadata['missing_values']:,}

## Feature Engineering Summary

{metadata['n_engineered_features']} new feature(s) were engineered from the raw transaction data. Categorical encoding was intentionally not applied at this stage; it is deferred to model training/inference.

## Engineered Features Added

{engineered_features_list}

## Output Dataset

`{metadata['output_dataset_path']}`

## Processing Timestamp

{metadata['execution_timestamp']}
"""


def _save_report(report: str, path: Path = REPORT_PATH) -> Path:
    """Write the Markdown preprocessing report to disk.

    Args:
        report: The report content, as Markdown text.
        path: Destination file path. Defaults to the configured report
            path.

    Returns:
        The path the report was written to.

    Raises:
        OSError: If the file cannot be written.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")
    except OSError as exc:
        raise OSError(f"Failed to write preprocessing report to {path}: {exc}") from exc

    logger.info("Saved preprocessing report to %s.", path)
    return path


def run_pipeline() -> pd.DataFrame:
    """Run the full preprocessing pipeline end-to-end.

    Sequence:
        1. Load the raw dataset.
        2. Validate required columns.
        3. Generate a cleaning summary (missing values, duplicates).
        4. Engineer additional features.
        5. Save the engineered dataset to ``data/processed``.

    On success, a JSON metadata file and Markdown report summarizing the
    run are written to ``reports/``. No categorical encoding is performed;
    that is deferred to model training/inference.

    Returns:
        The final engineered DataFrame, as written to disk.

    Raises:
        FileNotFoundError: If the raw dataset cannot be found.
        ValueError: If the raw dataset fails column validation.
    """
    logger.info("=== Preprocessing pipeline started ===")
    timestamp = datetime.now(timezone.utc).isoformat()

    logger.info("Step 1/5: Loading raw dataset.")
    df = load_raw_dataset()

    logger.info("Step 2/5: Validating schema.")
    validate_required_columns(df)

    logger.info("Step 3/5: Generating cleaning summary.")
    cleaning_summary = generate_cleaning_summary(df)

    logger.info("Step 4/5: Engineering features.")
    columns_before = df.columns.tolist()
    df = engineer_features(df)
    engineered_feature_names = [col for col in df.columns if col not in columns_before]

    logger.info("Step 5/5: Saving engineered dataset.")
    output_path = save_engineered_dataset(df)

    logger.info("Generating preprocessing metadata and report.")
    metadata = _build_metadata(cleaning_summary, engineered_feature_names, df, output_path, timestamp)
    _save_metadata(metadata)
    _save_report(_build_report(metadata))

    logger.info("=== Preprocessing pipeline finished. Output: %s ===", output_path)
    return df


def _configure_logging() -> None:
    """Configure root logging to write to both the console and a log file.

    The log file (``reports/logs/preprocessing.log``) is appended to
    rather than overwritten, so repeated pipeline runs accumulate a full
    history. Both handlers share the same log format.
    """
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)

    logging.basicConfig(level=logging.INFO, handlers=[console_handler, file_handler])


def main() -> None:
    """CLI entry point: configure logging and run the pipeline."""
    _configure_logging()
    run_pipeline()


if __name__ == "__main__":
    main()
