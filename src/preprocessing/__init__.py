"""Reusable preprocessing package for the fraud detection DSS.

Exposes the main building blocks of the preprocessing workflow so callers
can do ``from src.preprocessing import load_raw_dataset`` instead of
reaching into individual modules.
"""

from src.preprocessing.cleaning import (
    check_duplicates,
    check_missing_values,
    generate_cleaning_summary,
    validate_required_columns,
)
from src.preprocessing.encoding import encode_categorical_columns
from src.preprocessing.feature_engineering import engineer_features
from src.preprocessing.loader import (
    load_engineered_dataset,
    load_raw_dataset,
    save_engineered_dataset,
)
from src.preprocessing.pipeline import run_pipeline

__all__ = [
    "load_raw_dataset",
    "load_engineered_dataset",
    "save_engineered_dataset",
    "validate_required_columns",
    "check_missing_values",
    "check_duplicates",
    "generate_cleaning_summary",
    "engineer_features",
    "encode_categorical_columns",
    "run_pipeline",
]
