"""Abstract base class for all fraud detection model wrappers.

Defines the common interface (`fit`, `predict`, `predict_proba`, `save`,
`load`) that :mod:`src.models.trainer` and :mod:`src.models.evaluator`
rely on, so every model can be trained, evaluated, and persisted through
the exact same shared workflow regardless of the underlying algorithm.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Common wrapper interface around a scikit-learn-compatible estimator.

    Subclasses build a specific estimator (e.g. `LogisticRegression`,
    `RandomForestClassifier`) in :meth:`_build_estimator` and inherit
    uniform `fit`/`predict`/`predict_proba`/`save`/`load` behavior. This
    lets :mod:`src.models.trainer` train, evaluate, and persist any
    registered model polymorphically.

    Attributes:
        name: Short identifier used for logging, the model registry, and
            the persisted filename.
        uses_resampling: Whether the shared training workflow should apply
            SMOTE to this model's training data. Supervised classifiers
            default to ``True``; unsupervised/anomaly-detection models
            (e.g. Isolation Forest) should set this to ``False``, since
            resampling a fitting set intended to represent "normal"
            behavior would undermine the anomaly-detection premise.
        model: The underlying fitted (or unfitted) estimator instance.
    """

    def __init__(self, name: str, uses_resampling: bool = True) -> None:
        self.name = name
        self.uses_resampling = uses_resampling
        self.model: Any = self._build_estimator()

    @abstractmethod
    def _build_estimator(self) -> Any:
        """Instantiate and return the underlying estimator.

        Called once during ``__init__``. Subclasses should read
        hyperparameters from instance attributes set *before* calling
        ``super().__init__()``.
        """

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "BaseModel":
        """Fit the underlying estimator on training data.

        Args:
            X_train: Training feature matrix.
            y_train: Training target vector.

        Returns:
            ``self``, to allow chaining.
        """
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict binary fraud labels (0 = legitimate, 1 = fraud).

        Args:
            X: Feature matrix to score.

        Returns:
            An array of predicted binary labels.
        """
        return np.asarray(self.model.predict(X))

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """Predict a continuous fraud score/probability, if supported.

        Args:
            X: Feature matrix to score.

        Returns:
            An array of positive-class probabilities, or ``None`` if the
            underlying estimator does not expose `predict_proba`.
        """
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        return None

    def save(self, path: Path) -> Path:
        """Persist this model wrapper (including its fitted estimator) to disk.

        Args:
            path: Destination file path. Its parent directory is created
                automatically if it does not already exist.

        Returns:
            The path the model was written to.

        Raises:
            OSError: If the file cannot be written.
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self, path)
        except OSError as exc:
            raise OSError(f"Failed to save model '{self.name}' to {path}: {exc}") from exc

        logger.info("Saved model '%s' to %s.", self.name, path)
        return path

    @staticmethod
    def load(path: Path) -> "BaseModel":
        """Load a previously persisted model wrapper from disk.

        Args:
            path: Location of the saved ``.joblib`` file.

        Returns:
            The restored `BaseModel` subclass instance, ready for
            `predict`/`predict_proba`.

        Raises:
            FileNotFoundError: If no file exists at ``path``.
        """
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at: {path}")
        model = joblib.load(path)
        logger.info("Loaded model '%s' from %s.", getattr(model, "name", "unknown"), path)
        return model
