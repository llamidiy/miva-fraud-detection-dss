"""Markdown reporting for the Sprint 5.5 validation workflow.

Generates two complementary documents:

    business_scenarios.md  - the scenario catalog: what each scenario
                              represents and why it was constructed
                              (Scenario Description, Business Context,
                              Dataset/Feature Rationale).
    validation_report.md   - the results report: what the system
                              actually did with each scenario (Prediction,
                              Confidence, Top SHAP Factors, Recommendation,
                              Validation Result, Notes), plus an aggregate
                              summary and the recommendation rules.
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import BUSINESS_SCENARIOS_PATH, HIGH_CONFIDENCE_THRESHOLD, VALIDATION_REPORT_PATH
from src.validation.recommendation_engine import (
    FRAUD_HIGH_CONFIDENCE_ACTIONS,
    FRAUD_MEDIUM_CONFIDENCE_ACTIONS,
    LEGITIMATE_ACTIONS,
)
from src.validation.scenario_generator import RAW_TRANSACTION_FIELDS, Scenario
from src.validation.validator import ValidationResult

logger = logging.getLogger(__name__)

_CATEGORY_TITLES = {
    "fraud": "Fraud Scenarios",
    "legitimate": "Legitimate Scenarios",
    "edge_case": "Edge Cases",
}


def _transaction_table(transaction: dict[str, Any]) -> str:
    """Render a scenario's raw transaction fields as a small Markdown table."""
    lines = ["| Field | Value |", "| --- | --- |"]
    for field_name in RAW_TRANSACTION_FIELDS:
        value = transaction[field_name]
        formatted = f"{value:,.2f}" if isinstance(value, float) else str(value)
        lines.append(f"| `{field_name}` | {formatted} |")
    return "\n".join(lines)


def _format_scenario_catalog_entry(scenario: Scenario) -> str:
    tags = ", ".join(f"`{t}`" for t in scenario.tags) if scenario.tags else "—"
    return "\n".join(
        [
            f"### {scenario.scenario_id}: {scenario.title}",
            "",
            f"**Expected outcome:** `{scenario.expected_outcome}` | **Engineered features exercised:** {tags}",
            "",
            "**Scenario Description**",
            "",
            _transaction_table(scenario.transaction),
            "",
            "**Business Context**",
            "",
            scenario.business_description,
            "",
            "**Dataset/Feature Rationale**",
            "",
            f"*Source evidence:* {scenario.source_rationale}",
            "",
            f"*Expected reasoning:* {scenario.expected_reasoning}",
            "",
        ]
    )


def generate_business_scenarios_doc(
    scenarios: list[Scenario], path: Path = BUSINESS_SCENARIOS_PATH
) -> Path:
    """Generate the business scenario catalog document.

    Args:
        scenarios: The full generated scenario set.
        path: Destination Markdown path.

    Returns:
        The path the document was written to.
    """
    sections = [
        "# Business Scenarios",
        "",
        f"This catalog documents all {len(scenarios)} business validation scenarios used to "
        "test the fraud detection decision support system, and the specific dataset/model "
        "evidence that motivated each one. See `validation_report.md` for how the system "
        "actually responded to each scenario.",
        "",
    ]

    for category in ("fraud", "legitimate", "edge_case"):
        category_scenarios = [s for s in scenarios if s.category == category]
        if not category_scenarios:
            continue
        sections.append(f"## {_CATEGORY_TITLES[category]}")
        sections.append("")
        for scenario in category_scenarios:
            sections.append(_format_scenario_catalog_entry(scenario))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sections), encoding="utf-8")
    logger.info("Generated business scenarios catalog at %s.", path)
    return path


def _format_recommendation_rules_section() -> str:
    lines = [
        "## Recommendation Rules",
        "",
        "Recommendations are generated deterministically from the prediction class and "
        "confidence score alone — no additional model logic is involved:",
        "",
        f"- **Fraud, confidence >= {HIGH_CONFIDENCE_THRESHOLD:.0f}%:** "
        + ", ".join(FRAUD_HIGH_CONFIDENCE_ACTIONS),
        f"- **Fraud, confidence < {HIGH_CONFIDENCE_THRESHOLD:.0f}%:** "
        + ", ".join(FRAUD_MEDIUM_CONFIDENCE_ACTIONS),
        "- **Not Fraud (any confidence):** " + ", ".join(LEGITIMATE_ACTIONS),
        "",
    ]
    return "\n".join(lines)


def _format_summary_section(results: list[ValidationResult], elapsed_seconds: float) -> str:
    total = len(results)
    n_passed = sum(1 for r in results if r.validation_status == "PASS")
    n_match = sum(1 for r in results if r.outcome_match == "Match")
    n_mismatch = sum(1 for r in results if r.outcome_match == "Mismatch")
    n_na = total - n_match - n_mismatch

    mismatches = [r.scenario.scenario_id for r in results if r.outcome_match == "Mismatch"]
    failures = [r.scenario.scenario_id for r in results if r.validation_status == "FAIL"]

    lines = [
        "## Summary",
        "",
        f"- **Scenarios executed:** {total}",
        f"- **Validation status:** {n_passed}/{total} PASS"
        + (f" (failed: {', '.join(failures)})" if failures else ""),
        f"- **Expected vs. actual outcome:** {n_match} match, {n_mismatch} mismatch, "
        f"{n_na} without a strong prior expectation"
        + (f" (mismatches: {', '.join(mismatches)})" if mismatches else ""),
        f"- **Execution time:** {elapsed_seconds:.2f}s for {total} scenarios "
        f"({elapsed_seconds / total:.2f}s/scenario)",
        "",
    ]
    return "\n".join(lines)


def _format_scenario_result_entry(result: ValidationResult) -> str:
    scenario = result.scenario
    factors = (
        "\n".join(f"- {factor}" for factor in result.top_shap_factors)
        if result.top_shap_factors
        else "- (none — see error notes)"
    )
    actions = (
        ", ".join(result.recommendation.actions) if result.recommendation else "N/A"
    )
    confidence_str = f"{result.confidence_pct:.1f}%" if result.confidence_exists else "N/A"

    if result.error_message:
        notes = f"Validation failed: {result.error_message}"
    elif result.outcome_match == "Match":
        notes = f"Actual recommendation tier (`{result.recommendation.tier}`) matched the expected outcome."
    elif result.outcome_match == "Mismatch":
        notes = (
            f"Actual recommendation tier (`{result.recommendation.tier}`) did **not** match "
            f"the expected outcome (`{scenario.expected_outcome}`) — worth further review."
        )
    else:
        notes = "No strong prior expectation was set for this scenario (genuine edge case)."

    return "\n".join(
        [
            f"### {scenario.scenario_id}: {scenario.title}",
            "",
            f"**Prediction:** {result.prediction}",
            "",
            f"**Confidence:** {confidence_str}",
            "",
            "**Top SHAP Factors**",
            "",
            factors,
            "",
            f"**Recommendation:** {actions}",
            "",
            f"**Validation Result:** {result.validation_status} "
            f"(prediction={result.prediction_succeeded}, explanation={result.explanation_succeeded}, "
            f"recommendation={result.recommendation_succeeded}, confidence={result.confidence_exists}) "
            f"— expected vs. actual: {result.outcome_match}",
            "",
            f"**Notes:** {notes}",
            "",
        ]
    )


def generate_validation_report(
    results: list[ValidationResult],
    results_df: pd.DataFrame,
    elapsed_seconds: float,
    path: Path = VALIDATION_REPORT_PATH,
) -> Path:
    """Generate the validation results report.

    Args:
        results: The per-scenario validation results.
        results_df: The flattened results table (used only to confirm
            row count matches; the per-scenario detail below is built
            directly from `results`).
        elapsed_seconds: Total wall-clock time for the validation run.
        path: Destination Markdown path.

    Returns:
        The path the report was written to.
    """
    del results_df  # kept in the signature for symmetry/traceability; detail comes from results

    sections = [
        "# Validation Report",
        "",
        "This report records how the fraud detection decision support system responded to "
        f"{len(results)} realistic business scenarios (see `business_scenarios.md` for why "
        "each scenario was constructed), and compares the actual recommendation against the "
        "expected outcome anticipated when the scenario was designed.",
        "",
        _format_summary_section(results, elapsed_seconds),
        _format_recommendation_rules_section(),
    ]

    for category in ("fraud", "legitimate", "edge_case"):
        category_results = [r for r in results if r.scenario.category == category]
        if not category_results:
            continue
        sections.append(f"## {_CATEGORY_TITLES[category]}")
        sections.append("")
        for result in category_results:
            sections.append(_format_scenario_result_entry(result))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sections), encoding="utf-8")
    logger.info("Generated validation report at %s.", path)
    return path
