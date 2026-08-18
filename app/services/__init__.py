"""Service layer: the only part of the app allowed to talk to `src.*`.

Pages and components must call these services (and `app.state`)
instead of importing from `src.*` directly. Every service function
returns a simple, UI-friendly dataclass — never a raw backend object
(DataFrame, `shap.Explanation`, the model itself).
"""

from services.explainability_service import (
    FeatureImportanceRow,
    LocalExplanation,
    ShapSummary,
    get_champion_explainer,
    get_feature_importance,
    get_local_explanation,
    get_shap_summary,
    get_waterfall,
)
from services.metrics_service import ModelMetrics, get_model_metrics
from services.prediction_service import (
    BatchPredictionResult,
    BatchTransactionResult,
    PredictionResult,
    predict_batch,
    predict_transaction,
)
from services.report_service import (
    ReportContent,
    get_business_scenarios_report,
    get_executive_summary,
    get_validation_report,
)
from services.validation_service import ValidationSummary, get_validation_summary

__all__ = [
    "FeatureImportanceRow",
    "LocalExplanation",
    "ShapSummary",
    "get_champion_explainer",
    "get_feature_importance",
    "get_local_explanation",
    "get_shap_summary",
    "get_waterfall",
    "ModelMetrics",
    "get_model_metrics",
    "BatchPredictionResult",
    "BatchTransactionResult",
    "PredictionResult",
    "predict_batch",
    "predict_transaction",
    "ReportContent",
    "get_business_scenarios_report",
    "get_executive_summary",
    "get_validation_report",
    "ValidationSummary",
    "get_validation_summary",
]
