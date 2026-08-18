"""XGBoost model wrapper."""

from typing import Any, Optional

from src.config import RANDOM_STATE
from src.models.base_model import BaseModel

try:
    from xgboost import XGBClassifier
except Exception as exc:  # pragma: no cover - environment-dependent
    # xgboost's compiled extension can fail to import for environment
    # reasons (e.g. missing libomp on macOS) even when the package itself
    # is installed. Deferring the failure to instantiation time (rather
    # than letting it crash this module's import) keeps the rest of the
    # framework usable when only XGBoost is unavailable.
    XGBClassifier = None
    _XGBOOST_IMPORT_ERROR: Optional[Exception] = exc
else:
    _XGBOOST_IMPORT_ERROR = None


class XGBoostModel(BaseModel):
    """Gradient-boosted tree classifier (XGBoost) for fraud detection.

    Typically the strongest tabular baseline: captures non-linear
    interactions and handles unscaled numeric features well.

    Args:
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth per boosting round.
        learning_rate: Step size shrinkage applied at each boosting round.
        scale_pos_weight: Balances positive/negative class weighting.
            Defaults to ``None`` (i.e. 1, no reweighting), since the
            shared training workflow already balances classes via SMOTE
            before fitting.
        random_state: Seed for reproducibility.
        n_jobs: Number of parallel threads. ``-1`` uses all available
            cores.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        scale_pos_weight: Optional[float] = None,
        random_state: int = RANDOM_STATE,
        n_jobs: int = -1,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        super().__init__(name="xgboost")

    def _build_estimator(self) -> Any:
        if XGBClassifier is None:
            raise RuntimeError(
                "xgboost could not be imported, so the 'xgboost' model is "
                "unavailable. On macOS this is usually caused by a missing "
                "OpenMP runtime; try `brew install libomp`. "
                f"Original error: {_XGBOOST_IMPORT_ERROR}"
            ) from _XGBOOST_IMPORT_ERROR
        return XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            scale_pos_weight=self.scale_pos_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            eval_metric="logloss",
        )
