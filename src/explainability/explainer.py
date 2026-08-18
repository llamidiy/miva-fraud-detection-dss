"""SHAP-based explainability engine for the champion XGBoost fraud model.

Wraps the existing, frozen Sprint 4/4.5 XGBoost deployment artifacts
(model, encoder, metadata, feature schema — loaded via
`src.models.predictor.load_model_artifacts`) in a single reusable SHAP
`TreeExplainer`. No model is retrained or modified here; predictions go
through the existing predictor's public `predict`/`predict_proba`, so
scoring behavior is guaranteed identical to production.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
import shap

from src.config import CHAMPION_MODEL_NAME
from src.models.predictor import ModelArtifacts, load_model_artifacts, predict, predict_proba
from src.preprocessing.encoding import encode_with_fitted_encoder

logger = logging.getLogger(__name__)

#: Index of the positive ("fraud") class within XGBoost's binary SHAP output.
_POSITIVE_CLASS_INDEX = 1


class FraudExplainer:
    """A reusable SHAP explainer built on top of the champion model's artifacts.

    Attributes:
        model_name: Registered model name (defaults to the champion
            model, `CHAMPION_MODEL_NAME`).
        artifacts: The loaded `ModelArtifacts` bundle (model, encoder,
            metadata, feature schema).
        shap_explainer: A single `shap.TreeExplainer` built once and
            reused for every call.
    """

    def __init__(self, model_name: str = CHAMPION_MODEL_NAME) -> None:
        self.model_name = model_name
        self.artifacts: ModelArtifacts = load_model_artifacts(model_name)
        self.shap_explainer = shap.TreeExplainer(self.artifacts.model.model)
        logger.info("Initialized SHAP TreeExplainer for champion model '%s'.", model_name)

    def encode(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        """Encode raw (pre-encoding) feature data for this model.

        Uses the model's own persisted encoder and training-time column
        order, mirroring `src.models.predictor`'s internal preparation
        step (kept separate here since that step is private to the
        frozen predictor module).

        Args:
            X_raw: Raw feature data, including the categorical `type`
                column.

        Returns:
            The encoded feature matrix, columns in training-time order.

        Raises:
            ValueError: If `X_raw` is missing any required column.
        """
        expected = self.artifacts.feature_schema["raw_feature_names"]
        missing = [col for col in expected if col not in X_raw.columns]
        if missing:
            raise ValueError(f"Input is missing required columns: {missing}")

        encoded = encode_with_fitted_encoder(X_raw[expected], self.artifacts.encoder)
        return encoded[self.artifacts.feature_schema["encoded_feature_order"]]

    def get_shap_values(self, X_raw: pd.DataFrame) -> shap.Explanation:
        """Compute SHAP values for raw feature data.

        Normalizes XGBoost's binary-classification SHAP output to a
        consistent 2D-per-sample `Explanation` (values shaped
        `(n_samples, n_features)`), selecting the positive ("fraud")
        class if the underlying explainer returns a per-class axis, so
        downstream callers never need to handle that shape themselves.

        Args:
            X_raw: Raw feature data to explain.

        Returns:
            A `shap.Explanation` for the positive (fraud) class.
        """
        X_encoded = self.encode(X_raw)
        explanation = self.shap_explainer(X_encoded)

        if explanation.values.ndim == 3:
            base_values = explanation.base_values
            base_values = (
                base_values[:, _POSITIVE_CLASS_INDEX] if np.ndim(base_values) == 2 else base_values
            )
            explanation = shap.Explanation(
                values=explanation.values[:, :, _POSITIVE_CLASS_INDEX],
                base_values=base_values,
                data=explanation.data,
                feature_names=explanation.feature_names,
            )

        logger.info("Computed SHAP values for %d row(s).", len(X_raw))
        return explanation

    def get_shap_values_array(self, X_raw: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        """Compute SHAP values as a plain array, for legacy SHAP plot functions.

        Args:
            X_raw: Raw feature data to explain.

        Returns:
            A ``(values, X_encoded)`` tuple: the SHAP values as a 2D
            NumPy array `(n_samples, n_features)`, and the corresponding
            encoded feature matrix used to compute them.
        """
        explanation = self.get_shap_values(X_raw)
        X_encoded = self.encode(X_raw)
        return explanation.values, X_encoded

    def predict_transaction(self, X_raw: pd.DataFrame) -> dict[str, Any]:
        """Predict the fraud label and probability for one or more raw transactions.

        Reuses the existing predictor's public `predict`/`predict_proba`
        so scoring is identical to production.

        Args:
            X_raw: Raw feature data for one or more transactions.

        Returns:
            A dictionary with ``prediction`` (array of 0/1 labels) and
            ``fraud_probability`` (array of scores, or ``None`` if
            unsupported).
        """
        return {
            "prediction": predict(self.artifacts, X_raw),
            "fraud_probability": predict_proba(self.artifacts, X_raw),
        }

    def explain_transaction(self, X_raw: pd.DataFrame) -> dict[str, Any]:
        """Produce a full prediction + SHAP explanation for a single transaction.

        Args:
            X_raw: Raw feature data for exactly one transaction (a
                single-row DataFrame).

        Returns:
            A dictionary with ``prediction`` (int), ``fraud_probability``
            (float or ``None``), and ``shap_explanation`` (a single-row
            `shap.Explanation`).

        Raises:
            ValueError: If `X_raw` does not contain exactly one row.
        """
        if len(X_raw) != 1:
            raise ValueError(f"explain_transaction expects exactly one row, got {len(X_raw)}.")

        outcome = self.predict_transaction(X_raw)
        fraud_probability = outcome["fraud_probability"]

        return {
            "prediction": int(outcome["prediction"][0]),
            "fraud_probability": float(fraud_probability[0]) if fraud_probability is not None else None,
            "shap_explanation": self.get_shap_values(X_raw)[0],
        }
