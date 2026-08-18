"""Executes every business scenario end-to-end and orchestrates reporting.

Entry point for the whole Sprint 5.5 validation workflow: generates
scenarios, runs each through `validator.validate_scenario`, saves the
results table, and triggers the Markdown report generation. Intended to
be run as a script (``python -m src.validation.scenario_runner``) or
imported and called via `run_validation_workflow`.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import SCENARIO_RESULTS_PATH, SHAP_SAMPLE_PATH, VALIDATION_LOG_PATH
from src.explainability.explainer import FraudExplainer
from src.utils.logger import configure_logging
from src.validation.scenario_generator import Scenario, generate_scenarios
from src.validation.validator import ValidationResult, validate_scenario

logger = logging.getLogger(__name__)


def run_all_scenarios(
    explainer: Optional[FraudExplainer] = None,
    scenarios: Optional[list[Scenario]] = None,
) -> list[ValidationResult]:
    """Run every scenario through the full validation pipeline.

    Args:
        explainer: An initialized `FraudExplainer`. Built fresh (loading
            the champion model) if not provided.
        scenarios: Scenarios to run. Defaults to the full generated set
            (see `~src.validation.scenario_generator.generate_scenarios`).

    Returns:
        One `ValidationResult` per scenario, in the same order.
    """
    explainer = explainer or FraudExplainer()
    scenarios = scenarios if scenarios is not None else generate_scenarios()
    reference_sample = pd.read_csv(SHAP_SAMPLE_PATH)

    logger.info("Running %d scenarios through the validation pipeline.", len(scenarios))
    results = [validate_scenario(explainer, scenario, reference_sample) for scenario in scenarios]

    n_passed = sum(1 for r in results if r.validation_status == "PASS")
    logger.info("Scenario execution complete: %d/%d passed validation.", n_passed, len(results))
    return results


def build_results_dataframe(results: list[ValidationResult]) -> pd.DataFrame:
    """Flatten validation results into a CSV-friendly comparison table.

    Args:
        results: The per-scenario validation results.

    Returns:
        A DataFrame with one row per scenario.
    """
    rows = []
    for r in results:
        rows.append(
            {
                "scenario_id": r.scenario.scenario_id,
                "title": r.scenario.title,
                "category": r.scenario.category,
                "expected_outcome": r.scenario.expected_outcome,
                "prediction": r.prediction,
                "confidence_pct": round(r.confidence_pct, 2) if r.confidence_exists else None,
                "top_shap_factors": "; ".join(r.top_shap_factors),
                "recommendation_tier": r.recommendation.tier if r.recommendation else "N/A",
                "recommendation_actions": (
                    "; ".join(r.recommendation.actions) if r.recommendation else ""
                ),
                "outcome_match": r.outcome_match,
                "prediction_succeeded": r.prediction_succeeded,
                "explanation_succeeded": r.explanation_succeeded,
                "recommendation_succeeded": r.recommendation_succeeded,
                "confidence_exists": r.confidence_exists,
                "validation_status": r.validation_status,
                "error_message": r.error_message or "",
            }
        )
    return pd.DataFrame(rows)


def save_scenario_results(df: pd.DataFrame, path: Path = SCENARIO_RESULTS_PATH) -> Path:
    """Persist the scenario results table to disk as CSV.

    Args:
        df: The results DataFrame, as built by `build_results_dataframe`.
        path: Destination CSV path.

    Returns:
        The path the table was written to.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved scenario results table (%d scenarios) to %s.", len(df), path)
    return path


def run_validation_workflow() -> dict[str, Path]:
    """Run the full Sprint 5.5 validation workflow end-to-end.

    Generates scenarios, validates each through the frozen predictor and
    explainability layer, saves the results table, and generates both
    Markdown reports. No model is retrained or modified.

    Returns:
        A dict mapping each generated artifact's name to its path.
    """
    logger.info("=== Validation workflow started ===")
    start = time.perf_counter()

    scenarios = generate_scenarios()
    explainer = FraudExplainer()
    results = run_all_scenarios(explainer=explainer, scenarios=scenarios)

    results_df = build_results_dataframe(results)
    csv_path = save_scenario_results(results_df)

    # Imported here (not at module scope) to avoid a circular import with
    # report_generator, which itself calls back into this module's main().
    from src.validation.report_generator import (
        generate_business_scenarios_doc,
        generate_validation_report,
    )

    business_path = generate_business_scenarios_doc(scenarios)

    elapsed = time.perf_counter() - start
    validation_path = generate_validation_report(results, results_df, elapsed)

    logger.info("=== Validation workflow finished in %.2fs. ===", elapsed)
    return {
        "scenario_results_csv": csv_path,
        "business_scenarios_md": business_path,
        "validation_report_md": validation_path,
    }


def main() -> None:
    """CLI entry point: configure logging and run the full validation workflow."""
    configure_logging(VALIDATION_LOG_PATH)
    run_validation_workflow()


if __name__ == "__main__":
    main()
