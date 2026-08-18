"""Reusable inference utilities for trained fraud detection models.

The caller only specifies a model name. `load_model_artifacts` locates
that model's directory under `models/` and loads everything needed to
run inference — the fitted model, its fitted categorical encoder, its
training metadata, and its feature schema — into a single bundle.
`predict`/`predict_proba` then take that bundle plus *raw*
(pre-encoding) feature data, validate it against the feature schema,
encode it with the model's own persisted encoder, and reorder columns to
match what the model was trained on, before scoring.

This supersedes Sprint 4's `predictor.py`, which required the caller to
already have the exact fitted encoder used at training time — an
encoder that was never actually persisted, so real inference on new raw
data wasn't possible. Every artifact needed for inference now lives
inside `models/<model_name>/`, making each model directory self-contained.
"""

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.models.artifacts import load_encoder, load_feature_schema, load_metadata
from src.models.base_model import BaseModel
from src.models.model_registry import (
    get_encoder_path,
    get_feature_schema_path,
    get_metadata_path,
    get_model_path,
)
from src.preprocessing.encoding import encode_with_fitted_encoder

logger = logging.getLogger(__name__)


@dataclass
class ModelArtifacts:
    """A fully loaded, ready-to-use bundle of a trained model's deployment artifacts.

    Attributes:
        name: Registered model name.
        model: The fitted model wrapper.
        encoder: The fitted categorical encoder used at training time.
        metadata: The model's training metadata record.
        feature_schema: The model's feature schema, used to validate and
            prepare raw inference input.
    """

    name: str
    model: BaseModel
    encoder: Any
    metadata: dict[str, Any]
    feature_schema: dict[str, Any]


def load_model_artifacts(model_name: str) -> ModelArtifacts:
    """Load a model and every artifact needed to run inference with it.

    Args:
        model_name: Registered model name (see
            :func:`~src.models.model_registry.list_available_models`).

    Returns:
        A `ModelArtifacts` bundle containing the fitted model, its fitted
        encoder, its training metadata, and its feature schema.

    Raises:
        ValueError: If `model_name` is not registered.
        FileNotFoundError: If any expected artifact file is missing.
    """
    model = BaseModel.load(get_model_path(model_name))
    encoder = load_encoder(get_encoder_path(model_name))
    metadata = load_metadata(get_metadata_path(model_name))
    feature_schema = load_feature_schema(get_feature_schema_path(model_name))

    logger.info("Loaded full artifact bundle for '%s'.", model_name)
    return ModelArtifacts(
        name=model_name,
        model=model,
        encoder=encoder,
        metadata=metadata,
        feature_schema=feature_schema,
    )


def _prepare_input(artifacts: ModelArtifacts, X: pd.DataFrame) -> pd.DataFrame:
    """Validate, encode, and reorder raw input to match training-time columns.

    Args:
        artifacts: A loaded `ModelArtifacts` bundle.
        X: Raw (pre-encoding) feature data.

    Returns:
        The encoded feature matrix, with columns in the exact order the
        model was trained on.

    Raises:
        ValueError: If `X` is missing any column the feature schema
            requires.
    """
    expected = artifacts.feature_schema["raw_feature_names"]
    missing = [col for col in expected if col not in X.columns]
    if missing:
        raise ValueError(
            f"Input is missing required columns for model '{artifacts.name}': {missing}"
        )

    encoded = encode_with_fitted_encoder(X[expected], artifacts.encoder)
    return encoded[artifacts.feature_schema["encoded_feature_order"]]


def predict(artifacts: ModelArtifacts, X: pd.DataFrame) -> np.ndarray:
    """Predict binary fraud labels (0 = legitimate, 1 = fraud) for raw input.

    Args:
        artifacts: A loaded `ModelArtifacts` bundle (see
            :func:`load_model_artifacts`).
        X: Raw (pre-encoding) feature data.

    Returns:
        An array of predicted binary labels.
    """
    return artifacts.model.predict(_prepare_input(artifacts, X))


def predict_proba(artifacts: ModelArtifacts, X: pd.DataFrame) -> Optional[np.ndarray]:
    """Predict a continuous fraud score/probability for raw input, if supported.

    Args:
        artifacts: A loaded `ModelArtifacts` bundle (see
            :func:`load_model_artifacts`).
        X: Raw (pre-encoding) feature data.

    Returns:
        An array of fraud scores, or `None` if the underlying model does
        not support probability/score output.
    """
    return artifacts.model.predict_proba(_prepare_input(artifacts, X))
