"""Logistic Regression model wrapper."""

from typing import Optional

from sklearn.linear_model import LogisticRegression

from src.config import RANDOM_STATE
from src.models.base_model import BaseModel


class LogisticRegressionModel(BaseModel):
    """Logistic Regression baseline classifier for fraud detection.

    A linear, highly interpretable baseline. `class_weight` defaults to
    ``None`` rather than ``"balanced"`` because the shared training
    workflow already balances classes via SMOTE before fitting; applying
    both would double-compensate for class imbalance.

    Args:
        C: Inverse of regularization strength (smaller = stronger
            regularization).
        max_iter: Maximum solver iterations. Raised above scikit-learn's
            default because the engineered features are unscaled and can
            slow convergence.
        class_weight: Passed through to `LogisticRegression`. Defaults to
            ``None``; set to ``"balanced"`` to combine with (or in place
            of) SMOTE-based resampling.
        random_state: Seed for reproducibility.
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: Optional[str] = None,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state
        super().__init__(name="logistic_regression")

    def _build_estimator(self) -> LogisticRegression:
        return LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.random_state,
        )
