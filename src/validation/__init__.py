"""System validation layer: business scenarios, edge cases, and end-to-end checks.

Exercises the frozen Sprint 3-5 preprocessing, model, and explainability
components against realistic business scenarios (see
`scenario_generator.py` and `edge_cases.py`), without retraining or
modifying any of them.
"""

from src.validation.recommendation_engine import Recommendation, generate_recommendation
from src.validation.report_generator import generate_business_scenarios_doc, generate_validation_report
from src.validation.scenario_generator import Scenario, cite_feature_evidence, generate_scenarios
from src.validation.scenario_runner import (
    build_results_dataframe,
    run_all_scenarios,
    run_validation_workflow,
    save_scenario_results,
)
from src.validation.validator import ValidationResult, build_transaction_dataframe, validate_scenario

__all__ = [
    "Scenario",
    "generate_scenarios",
    "cite_feature_evidence",
    "ValidationResult",
    "build_transaction_dataframe",
    "validate_scenario",
    "Recommendation",
    "generate_recommendation",
    "run_all_scenarios",
    "build_results_dataframe",
    "save_scenario_results",
    "run_validation_workflow",
    "generate_business_scenarios_doc",
    "generate_validation_report",
]
