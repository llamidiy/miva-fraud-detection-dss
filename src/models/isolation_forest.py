"""Isolation Forest model wrapper (unsupervised anomaly detection)."""

from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.config import RANDOM_STATE
from src.models.base_model import BaseModel


class IsolationForestModel(BaseModel):
    """Isolation Forest anomaly detector for fraud detection.

    Unlike the other wrappers, Isolation Forest is unsupervised: it is
    fit only on `X` (the target is accepted for interface consistency
    with the shared training workflow but ignored), and it isolates
    anomalies rather than learning a decision boundary between labeled
    classes. Because of this, `uses_resampling` is set to ``False`` — the
    shared training workflow skips SMOTE for this model, since
    artificially balancing the classes would distort what "normal"
    behavior looks like during fitting.

    scikit-learn's `IsolationForest` predicts ``-1`` for anomalies and
    ``1`` for normal points; this wrapper remaps that to the project's
    ``{0, 1}`` fraud-label convention (``1`` = fraud) so it is a drop-in
    replacement for the supervised models.

    Args:
        n_estimators: Number of isolation trees.
        contamination: Expected proportion of anomalies in the data.
            ``"auto"`` uses scikit-learn's default threshold heuristic;
            set to the known fraud prevalence (e.g. ``0.0013`` for
            PaySim) for a tighter, data-informed decision threshold.
        max_samples: Number of samples drawn to train each tree.
        random_state: Seed for reproducibility.
        n_jobs: Number of parallel jobs. ``-1`` uses all available cores.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        contamination: Union[str, float] = "auto",
        max_samples: Union[str, int, float] = "auto",
        random_state: int = RANDOM_STATE,
        n_jobs: int = -1,
    ) -> None:
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples
        self.random_state = random_state
        self.n_jobs = n_jobs
        super().__init__(name="isolation_forest", uses_resampling=False)

    def _build_estimator(self) -> IsolationForest:
        return IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "IsolationForestModel":
        """Fit the isolation forest on features only.

        Args:
            X_train: Training feature matrix.
            y_train: Accepted for interface consistency with
                :meth:`BaseModel.fit` but not used during fitting.

        Returns:
            ``self``, to allow chaining.
        """
        del y_train
        self.model.fit(X_train)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict binary fraud labels from the anomaly labels.

        Remaps scikit-learn's ``{-1, 1}`` anomaly labels to this
        project's ``{1, 0}`` fraud-label convention.

        Args:
            X: Feature matrix to score.

        Returns:
            An array of predicted binary labels (1 = fraud/anomaly).
        """
        raw_labels = self.model.predict(X)
        return np.where(raw_labels == -1, 1, 0)

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """Return a continuous anomaly score usable for ROC-AUC.

        `IsolationForest` has no `predict_proba`; `decision_function`
        returns higher values for more "normal" points, so it is negated
        here to produce a fraud-likeness score (higher = more anomalous).
        ROC-AUC is rank-based, so this monotonic transform does not
        affect the resulting score.

        Args:
            X: Feature matrix to score.

        Returns:
            An array of fraud-likeness scores.
        """
        return -self.model.decision_function(X)
