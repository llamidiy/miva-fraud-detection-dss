"""Central registry mapping model names to wrapper classes and artifact paths.

The single source of truth for which models exist and where their
persisted artifacts live, so :mod:`src.models.trainer` and
:mod:`src.models.predictor` never hard-code model classes or filenames
directly.

Each registered model gets its own directory under `MODEL_DIR`
(`models/<name>/`) holding four artifacts — the model, its fitted
encoder, its training metadata, and its feature schema — so a model
directory is fully self-contained for deployment. See
:mod:`src.models.artifacts` for the save/load logic behind each file.
"""

from pathlib import Path
from typing import Any, Type

from src.config import MODEL_DIR
from src.models.base_model import BaseModel
from src.models.isolation_forest import IsolationForestModel
from src.models.logistic_regression import LogisticRegressionModel
from src.models.random_forest import RandomForestModel
from src.models.xgboost_model import XGBoostModel

#: Maps a model name to its wrapper class.
MODEL_REGISTRY: dict[str, Type[BaseModel]] = {
    "logistic_regression": LogisticRegressionModel,
    "random_forest": RandomForestModel,
    "xgboost": XGBoostModel,
    "isolation_forest": IsolationForestModel,
}

#: Standard artifact filenames within each model's directory.
MODEL_FILENAME: str = "model.joblib"
ENCODER_FILENAME: str = "encoder.joblib"
METADATA_FILENAME: str = "metadata.json"
FEATURE_SCHEMA_FILENAME: str = "feature_schema.json"


def list_available_models() -> list[str]:
    """Return the names of all registered models."""
    return list(MODEL_REGISTRY.keys())


def get_model(name: str, **kwargs: Any) -> BaseModel:
    """Instantiate a registered model wrapper by name.

    Args:
        name: Registered model name (see :func:`list_available_models`).
        **kwargs: Hyperparameters forwarded to the model wrapper's
            constructor.

    Returns:
        A newly constructed, unfitted model wrapper instance.

    Raises:
        ValueError: If ``name`` is not registered.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available models: {list_available_models()}")
    return MODEL_REGISTRY[name](**kwargs)


def get_model_dir(name: str) -> Path:
    """Resolve the artifact directory for a registered model.

    Args:
        name: Registered model name (see :func:`list_available_models`).

    Returns:
        The directory under `MODEL_DIR` holding this model's artifacts
        (`model.joblib`, `encoder.joblib`, `metadata.json`,
        `feature_schema.json`).

    Raises:
        ValueError: If ``name`` is not registered.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available models: {list_available_models()}")
    return MODEL_DIR / name


def get_model_path(name: str) -> Path:
    """Resolve the `model.joblib` path for a registered model."""
    return get_model_dir(name) / MODEL_FILENAME


def get_encoder_path(name: str) -> Path:
    """Resolve the `encoder.joblib` path for a registered model."""
    return get_model_dir(name) / ENCODER_FILENAME


def get_metadata_path(name: str) -> Path:
    """Resolve the `metadata.json` path for a registered model."""
    return get_model_dir(name) / METADATA_FILENAME


def get_feature_schema_path(name: str) -> Path:
    """Resolve the `feature_schema.json` path for a registered model."""
    return get_model_dir(name) / FEATURE_SCHEMA_FILENAME
