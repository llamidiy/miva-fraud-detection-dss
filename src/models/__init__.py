"""Reusable model training/evaluation/persistence framework.

Exposes the main building blocks so callers can do
``from src.models import get_model, run_training`` instead of reaching
into individual modules.
"""

from src.models.base_model import BaseModel
from src.models.evaluator import evaluate_model, save_comparison_table
from src.models.isolation_forest import IsolationForestModel
from src.models.logistic_regression import LogisticRegressionModel
from src.models.model_registry import get_model, get_model_dir, get_model_path, list_available_models
from src.models.predictor import ModelArtifacts, load_model_artifacts, predict, predict_proba
from src.models.random_forest import RandomForestModel
from src.models.trainer import run_training
from src.models.xgboost_model import XGBoostModel

__all__ = [
    "BaseModel",
    "LogisticRegressionModel",
    "RandomForestModel",
    "XGBoostModel",
    "IsolationForestModel",
    "get_model",
    "get_model_dir",
    "get_model_path",
    "list_available_models",
    "evaluate_model",
    "save_comparison_table",
    "ModelArtifacts",
    "load_model_artifacts",
    "predict",
    "predict_proba",
    "run_training",
]
