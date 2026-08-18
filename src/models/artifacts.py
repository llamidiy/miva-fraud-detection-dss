"""Persistence helpers for per-model deployment artifacts.

Each registered model gets its own directory under `models/`
(`models/<model_name>/`) containing the artifacts needed to run
inference on new data without any other in-memory state:

    model.joblib          - the fitted model wrapper (see base_model.py)
    encoder.joblib         - the fitted categorical encoder
    metadata.json          - a record of how/when the model was trained
    feature_schema.json    - the input contract for inference

This module only handles the raw save/load I/O for the encoder,
metadata, and feature schema. `src.models.base_model.BaseModel` already
handles model save/load, and `src.models.trainer` decides what goes
into the metadata/schema dictionaries.
"""

import json
import logging
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger(__name__)


def save_encoder(encoder: Any, path: Path) -> Path:
    """Persist a fitted categorical encoder to disk via joblib.

    Args:
        encoder: A fitted encoder (e.g. a scikit-learn `OneHotEncoder`).
        path: Destination file path. Its parent directory is created
            automatically if it does not already exist.

    Returns:
        The path the encoder was written to.

    Raises:
        OSError: If the file cannot be written.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(encoder, path)
    except OSError as exc:
        raise OSError(f"Failed to save encoder to {path}: {exc}") from exc

    logger.info("Persisted fitted encoder to %s.", path)
    return path


def load_encoder(path: Path) -> Any:
    """Load a previously persisted fitted encoder from disk.

    Args:
        path: Location of the saved encoder ``.joblib`` file.

    Returns:
        The restored, fitted encoder.

    Raises:
        FileNotFoundError: If no file exists at ``path``.
    """
    if not path.exists():
        raise FileNotFoundError(f"Encoder file not found at: {path}")
    encoder = joblib.load(path)
    logger.info("Loaded fitted encoder from %s.", path)
    return encoder


def save_metadata(metadata: dict[str, Any], path: Path) -> Path:
    """Persist a model metadata dictionary to disk as JSON.

    Args:
        metadata: The metadata dictionary to persist.
        path: Destination file path. Its parent directory is created
            automatically if it does not already exist.

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
        raise OSError(f"Failed to save model metadata to {path}: {exc}") from exc

    logger.info("Generated model metadata at %s.", path)
    return path


def load_metadata(path: Path) -> dict[str, Any]:
    """Load a previously persisted model metadata dictionary from disk.

    Args:
        path: Location of the saved ``metadata.json`` file.

    Returns:
        The metadata dictionary.

    Raises:
        FileNotFoundError: If no file exists at ``path``.
    """
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found at: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_feature_schema(schema: dict[str, Any], path: Path) -> Path:
    """Persist a feature schema dictionary to disk as JSON.

    Args:
        schema: The feature schema dictionary to persist.
        path: Destination file path. Its parent directory is created
            automatically if it does not already exist.

    Returns:
        The path the feature schema was written to.

    Raises:
        OSError: If the file cannot be written.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2)
    except OSError as exc:
        raise OSError(f"Failed to save feature schema to {path}: {exc}") from exc

    logger.info("Created feature schema at %s.", path)
    return path


def load_feature_schema(path: Path) -> dict[str, Any]:
    """Load a previously persisted feature schema dictionary from disk.

    Args:
        path: Location of the saved ``feature_schema.json`` file.

    Returns:
        The feature schema dictionary.

    Raises:
        FileNotFoundError: If no file exists at ``path``.
    """
    if not path.exists():
        raise FileNotFoundError(f"Feature schema file not found at: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
