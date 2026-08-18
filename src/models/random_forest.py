"""Random Forest model wrapper."""

from typing import Optional

from sklearn.ensemble import RandomForestClassifier

from src.config import RANDOM_STATE
from src.models.base_model import BaseModel


class RandomForestModel(BaseModel):
    """Random Forest classifier for fraud detection.

    A bagged ensemble of decision trees, robust to unscaled/skewed
    features and able to capture non-linear interactions between
    engineered features.

    Args:
        n_estimators: Number of trees in the forest.
        max_depth: Maximum tree depth. ``None`` grows nodes until leaves
            are pure or contain fewer than `min_samples_split` samples.
        class_weight: Passed through to `RandomForestClassifier`. Defaults
            to ``None``; the shared training workflow already balances
            classes via SMOTE before fitting.
        random_state: Seed for reproducibility.
        n_jobs: Number of parallel jobs. ``-1`` uses all available cores.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        class_weight: Optional[str] = None,
        random_state: int = RANDOM_STATE,
        n_jobs: int = -1,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        super().__init__(name="random_forest")

    def _build_estimator(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
