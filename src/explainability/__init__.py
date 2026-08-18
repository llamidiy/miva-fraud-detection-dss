"""Explainability (XAI) layer for the champion XGBoost fraud model.

Builds SHAP-based global and local explanations on top of the existing,
frozen Sprint 4/4.5 model artifacts, without retraining or modifying
them.
"""

from src.explainability.explainer import FraudExplainer
from src.explainability.feature_importance import compute_feature_importance, run_feature_importance
from src.explainability.local_explanations import explain_single_transaction
from src.explainability.report_generator import (
    generate_model_interpretation_report,
    run_explainability_workflow,
)
from src.explainability.sampling import create_explainability_sample
from src.explainability.shap_analysis import run_shap_analysis

__all__ = [
    "FraudExplainer",
    "create_explainability_sample",
    "compute_feature_importance",
    "run_feature_importance",
    "run_shap_analysis",
    "explain_single_transaction",
    "generate_model_interpretation_report",
    "run_explainability_workflow",
]
