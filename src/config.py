"""Centralized project configuration.

All filesystem paths used across the fraud detection package are defined
here as :class:`pathlib.Path` objects, resolved relative to the project
root. Other modules should import from this file rather than constructing
paths independently, so the directory layout only needs to change in one
place.
"""

from pathlib import Path

# Project root (two levels up from this file: src/config.py -> project root)
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Data directories
RAW_DATA_DIR: Path = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR: Path = PROJECT_ROOT / "data" / "processed"

# Other project directories
MODELS_DIR: Path = PROJECT_ROOT / "models"
DATABASE_DIR: Path = PROJECT_ROOT / "database"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

# Dataset filenames
RAW_DATASET_FILENAME: str = "paysim.csv"
ENGINEERED_DATASET_FILENAME: str = "paysim_engineered.csv"

# Full dataset file paths
RAW_DATASET_PATH: Path = RAW_DATA_DIR / RAW_DATASET_FILENAME
ENGINEERED_DATASET_PATH: Path = PROCESSED_DATA_DIR / ENGINEERED_DATASET_FILENAME

# Preprocessing report filenames and paths
METADATA_FILENAME: str = "preprocessing_metadata.json"
REPORT_FILENAME: str = "preprocessing_report.md"
METADATA_PATH: Path = REPORTS_DIR / METADATA_FILENAME
REPORT_PATH: Path = REPORTS_DIR / REPORT_FILENAME

# Logging
LOG_DIR: Path = REPORTS_DIR / "logs"
LOG_FILENAME: str = "preprocessing.log"
LOG_PATH: Path = LOG_DIR / LOG_FILENAME

MODEL_TRAINING_LOG_FILENAME: str = "model_training.log"
MODEL_TRAINING_LOG_PATH: Path = LOG_DIR / MODEL_TRAINING_LOG_FILENAME

# Modeling constants (reused across Sprint 4 training/evaluation)
RANDOM_STATE: int = 42
TARGET_COLUMN: str = "isFraud"

TEST_SIZE: float = 0.20
VALIDATION_SIZE: float = 0.10

MODEL_DIR: Path = MODELS_DIR

# Model evaluation comparison table
MODEL_RESULTS_FILENAME: str = "model_results.csv"
MODEL_RESULTS_PATH: Path = REPORTS_DIR / MODEL_RESULTS_FILENAME

# Explainability (Sprint 5)
CHAMPION_MODEL_NAME: str = "xgboost"
EXPLAINABILITY_SAMPLE_SIZE: int = 10_000

EXPLAINABILITY_DATA_DIR: Path = PROJECT_ROOT / "data" / "explainability"
SHAP_SAMPLE_FILENAME: str = "shap_sample.csv"
SHAP_SAMPLE_PATH: Path = EXPLAINABILITY_DATA_DIR / SHAP_SAMPLE_FILENAME

FIGURES_DIR: Path = REPORTS_DIR / "figures"
SHAP_DIR: Path = REPORTS_DIR / "shap"

FEATURE_IMPORTANCE_FILENAME: str = "feature_importance.csv"
FEATURE_IMPORTANCE_PATH: Path = REPORTS_DIR / FEATURE_IMPORTANCE_FILENAME

SHAP_VALUES_FILENAME: str = "shap_values.joblib"
SHAP_VALUES_PATH: Path = SHAP_DIR / SHAP_VALUES_FILENAME

MODEL_INTERPRETATION_REPORT_FILENAME: str = "model_interpretation.md"
MODEL_INTERPRETATION_REPORT_PATH: Path = REPORTS_DIR / MODEL_INTERPRETATION_REPORT_FILENAME

EXPLAINABILITY_LOG_FILENAME: str = "explainability.log"
EXPLAINABILITY_LOG_PATH: Path = LOG_DIR / EXPLAINABILITY_LOG_FILENAME

# System validation (Sprint 5.5)
VALIDATION_DIR: Path = REPORTS_DIR / "validation"
VALIDATION_SCREENSHOTS_DIR: Path = VALIDATION_DIR / "screenshots"

SCENARIO_RESULTS_FILENAME: str = "scenario_results.csv"
SCENARIO_RESULTS_PATH: Path = VALIDATION_DIR / SCENARIO_RESULTS_FILENAME

VALIDATION_REPORT_FILENAME: str = "validation_report.md"
VALIDATION_REPORT_PATH: Path = VALIDATION_DIR / VALIDATION_REPORT_FILENAME

BUSINESS_SCENARIOS_FILENAME: str = "business_scenarios.md"
BUSINESS_SCENARIOS_PATH: Path = VALIDATION_DIR / BUSINESS_SCENARIOS_FILENAME

EXECUTIVE_SUMMARY_FILENAME: str = "executive_summary.md"
EXECUTIVE_SUMMARY_PATH: Path = VALIDATION_DIR / EXECUTIVE_SUMMARY_FILENAME

VALIDATION_LOG_FILENAME: str = "validation.log"
VALIDATION_LOG_PATH: Path = LOG_DIR / VALIDATION_LOG_FILENAME

# UI service layer (Sprint 6 Phase 2)
UI_INTEGRATION_LOG_FILENAME: str = "ui_integration.log"
UI_INTEGRATION_LOG_PATH: Path = LOG_DIR / UI_INTEGRATION_LOG_FILENAME

# Recommendation engine confidence tier: at/above this, a "Fraud" prediction is
# treated as high-confidence (flag/suspend/escalate); below it, medium-confidence
# (request verification/monitor). "Not Fraud" predictions are always the
# legitimate tier (approve/continue monitoring) regardless of confidence.
HIGH_CONFIDENCE_THRESHOLD: float = 90.0
