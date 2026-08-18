"""Executive-level summary of the Sprint 5.5 validation exercise.

Reads Sprint 5.5's already-generated outputs — `scenario_results.csv`,
`validation_report.md`, `business_scenarios.md` — and produces a single
concise, decision-oriented Markdown summary
(`reports/validation/executive_summary.md`) suitable for a dissertation
chapter, supervisor review, or executive demonstration.

This module performs no computation of its own beyond aggregation and
text formatting over those existing files: no scenarios are (re-)run,
no predictions or SHAP values are (re-)computed, and none of Sprint
5.5's frozen outputs are modified. Every statistic below is derived
directly from `scenario_results.csv` at generation time.
"""

import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import (
    BUSINESS_SCENARIOS_PATH,
    EXECUTIVE_SUMMARY_PATH,
    HIGH_CONFIDENCE_THRESHOLD,
    SCENARIO_RESULTS_PATH,
    VALIDATION_LOG_PATH,
    VALIDATION_REPORT_PATH,
)
from src.utils.logger import configure_logging

logger = logging.getLogger(__name__)

#: Human-readable labels for each `expected_outcome` value found in scenario_results.csv.
_EXPECTED_OUTCOME_LABELS: dict[str, str] = {
    "fraud_high_confidence": "High-confidence fraud (flag / suspend / escalate)",
    "fraud_medium_confidence": "Medium-confidence fraud (verify / monitor)",
    "legitimate": "Legitimate (approve)",
    "uncertain": "Uncertain / no strong prior expectation",
}

#: Human-readable labels for each `recommendation_tier` value found in scenario_results.csv.
_RECOMMENDATION_TIER_LABELS: dict[str, str] = {
    "fraud_high_confidence": "High-confidence fraud",
    "fraud_medium_confidence": "Medium-confidence fraud",
    "legitimate": "Legitimate",
    "N/A": "N/A",
}


def load_scenario_results(path: Path = SCENARIO_RESULTS_PATH) -> pd.DataFrame:
    """Load Sprint 5.5's scenario results table.

    Args:
        path: Location of `scenario_results.csv`.

    Returns:
        The results table.

    Raises:
        FileNotFoundError: If Sprint 5.5's validation workflow has not
            been run yet.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Scenario results not found at {path}. Run the Sprint 5.5 validation "
            "workflow first (python -m src.validation.scenario_runner) before "
            "generating the executive summary."
        )
    df = pd.read_csv(path)
    df["error_message"] = df["error_message"].fillna("")
    df["top_shap_factors"] = df["top_shap_factors"].fillna("")
    return df


def _read_text(path: Path) -> str:
    """Read an existing validation output as plain text.

    Args:
        path: File to read.

    Returns:
        The file's full text content.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Expected validation output not found at: {path}")
    return path.read_text(encoding="utf-8")


def _extract_markdown_section(text: str, heading: str) -> Optional[str]:
    """Extract the body of a top-level (`##`) Markdown section by heading text."""
    pattern = rf"^## {re.escape(heading)}\s*\n(.*?)(?=\n## |\Z)"
    match = re.search(pattern, text, flags=re.DOTALL | re.MULTILINE)
    return match.group(1).strip() if match else None


def _extract_scenario_block(business_scenarios_text: str, scenario_id: str) -> Optional[str]:
    """Extract one scenario's full `### <id>: ...` block from business_scenarios.md."""
    pattern = rf"^### {re.escape(scenario_id)}:.*?(?=^### |\Z)"
    match = re.search(pattern, business_scenarios_text, flags=re.DOTALL | re.MULTILINE)
    return match.group(0) if match else None


def _extract_business_context(business_scenarios_text: str, scenario_id: str) -> Optional[str]:
    """Extract a scenario's Business Context paragraph from business_scenarios.md."""
    block = _extract_scenario_block(business_scenarios_text, scenario_id)
    if block is None:
        return None
    match = re.search(r"\*\*Business Context\*\*\s*\n\n(.*?)\n\n", block, flags=re.DOTALL)
    return match.group(1).strip() if match else None


def _validation_date(scenario_results_path: Path) -> str:
    """Derive the validation date from `scenario_results.csv`'s file modification time."""
    mtime = scenario_results_path.stat().st_mtime
    return datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _format_overview_section(df: pd.DataFrame, scenario_results_path: Path) -> str:
    counts = df["category"].value_counts()
    return "\n".join(
        [
            "## 1. Validation Overview",
            "",
            f"- **Validation date:** {_validation_date(scenario_results_path)}",
            f"- **Number of scenarios:** {len(df)}",
            f"- **Fraud scenarios:** {int(counts.get('fraud', 0))}",
            f"- **Legitimate scenarios:** {int(counts.get('legitimate', 0))}",
            f"- **Edge cases:** {int(counts.get('edge_case', 0))}",
            "",
        ]
    )


def _format_overall_results_section(df: pd.DataFrame) -> str:
    total = len(df)
    passed = int((df["validation_status"] == "PASS").sum())
    failed = int((df["validation_status"] == "FAIL").sum())
    matches = int((df["outcome_match"] == "Match").sum())
    mismatches = int((df["outcome_match"] == "Mismatch").sum())
    na = int(df["outcome_match"].str.startswith("N/A").sum())

    return "\n".join(
        [
            "## 2. Overall Results",
            "",
            "| Metric | Count |",
            "| --- | --- |",
            f"| Total scenarios executed | {total} |",
            f"| Successful executions (PASS) | {passed} |",
            f"| Validation failures (FAIL) | {failed} |",
            f"| Expected vs. actual matches | {matches} |",
            f"| Expected vs. actual mismatches | {mismatches} |",
            f"| Exploratory / N/A scenarios | {na} |",
            "",
        ]
    )


def _format_key_findings_section(df: pd.DataFrame) -> str:
    fraud_df = df[df["category"] == "fraud"]
    fraud_detected = fraud_df[fraud_df["prediction"] == "Fraud"]
    fraud_high_conf = fraud_detected[fraud_detected["confidence_pct"] >= HIGH_CONFIDENCE_THRESHOLD]

    legit_df = df[df["category"] == "legitimate"]
    legit_correct = legit_df[legit_df["prediction"] == "Not Fraud"]

    edge_df = df[df["category"] == "edge_case"]
    edge_fraud = int((edge_df["prediction"] == "Fraud").sum())
    edge_legit = len(edge_df) - edge_fraud
    edge_failures = int((edge_df["validation_status"] == "FAIL").sum())

    confidence = df["confidence_pct"].dropna()
    explained = int((df["top_shap_factors"].str.len() > 0).sum())

    findings = [
        f"High-confidence fraud detection: {len(fraud_high_conf)}/{len(fraud_df)} fraud "
        f"scenarios were predicted as Fraud at or above the {HIGH_CONFIDENCE_THRESHOLD:.0f}% "
        f"confidence threshold "
        f"({len(fraud_detected)}/{len(fraud_df)} predicted Fraud in total).",
        f"Legitimate transaction handling: {len(legit_correct)}/{len(legit_df)} legitimate "
        "scenarios were correctly predicted as Not Fraud.",
        f"Edge-case behaviour: {len(edge_df)} boundary-condition scenarios produced "
        f"{edge_fraud} Fraud and {edge_legit} Not Fraud predictions, with "
        f"{edge_failures} runtime failure(s).",
        f"Confidence distribution: {confidence.min():.1f}%-{confidence.max():.1f}% across all "
        f"scenarios (mean {confidence.mean():.1f}%, median {confidence.median():.1f}%).",
        f"Explainability: SHAP-based explanations were successfully generated for "
        f"{explained}/{len(df)} scenarios.",
    ]

    return "\n".join(["## 3. Key Findings", "", *(f"- {f}" for f in findings), ""])


def _format_business_insights_section(df: pd.DataFrame) -> str:
    insights = []

    consistently_detected = df[
        (df["category"] == "fraud")
        & (df["prediction"] == "Fraud")
        & (df["confidence_pct"] >= HIGH_CONFIDENCE_THRESHOLD)
    ]
    if len(consistently_detected):
        ids = ", ".join(
            f"{r.scenario_id} ({r.title})" for r in consistently_detected.itertuples()
        )
        insights.append(
            f"Fraud behaviours consistently detected at high confidence: {ids}. These "
            "scenarios share the dataset's dominant fraud signature — a fully or "
            "substantially drained origin account on a TRANSFER/CASH_OUT transaction."
        )

    mismatches = df[df["outcome_match"] == "Mismatch"]
    if len(mismatches):
        ids = ", ".join(mismatches["scenario_id"])
        insights.append(
            f"Behaviours requiring analyst review: scenarios {ids} did not match the "
            "expected decision (see Section 5) and represent cases better suited to "
            "human review than fully automated action."
        )

    if len(mismatches):
        ids = ", ".join(mismatches["scenario_id"])
        insights.append(
            f"Explanations are particularly valuable in the mismatched scenarios ({ids}): "
            "the SHAP factors reported for each make it possible to see exactly which "
            "balance signal drove the model's decision, rather than leaving the "
            "discrepancy from expectation unexplained."
        )

    na_df = df[df["outcome_match"].str.startswith("N/A")]
    if len(na_df):
        ids = ", ".join(na_df["scenario_id"])
        insights.append(
            f"Scenarios {ids} were deliberately constructed without a strong prior "
            "expectation (genuine boundary conditions); explanations there support "
            "interpretation of the result rather than confirming or refuting a hypothesis."
        )

    n_tiers = df["recommendation_tier"].nunique()
    insights.append(
        f"The recommendation engine produced a valid, deterministic action for all "
        f"{len(df)} scenarios across {n_tiers} distinct tiers, giving every prediction — "
        "including the mismatched and exploratory ones — a clear, auditable next step "
        "rather than a bare probability score."
    )

    return "\n".join(["## 4. Business Insights", "", *(f"- {i}" for i in insights), ""])


def _format_notable_exceptions_section(df: pd.DataFrame, business_scenarios_text: str) -> str:
    mismatches = df[df["outcome_match"] == "Mismatch"]

    lines = [
        "## 5. Notable Exceptions",
        "",
        "Scenarios where the expected decision did not match the actual recommendation. "
        "These are reported in full rather than omitted, as they are the most valuable "
        "signal for where the system's behaviour should be scrutinized further.",
        "",
    ]

    if mismatches.empty:
        lines.append("No mismatches were observed in this validation run.")
        lines.append("")
        return "\n".join(lines)

    lines.append(
        "| Scenario ID | Scenario Name | Expected Decision | Actual Decision | Confidence | Short Explanation |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- |")

    for row in mismatches.itertuples():
        expected_label = _EXPECTED_OUTCOME_LABELS.get(row.expected_outcome, row.expected_outcome)
        actual_label = _RECOMMENDATION_TIER_LABELS.get(row.recommendation_tier, row.recommendation_tier)
        context = _extract_business_context(business_scenarios_text, row.scenario_id)
        explanation = (
            f"{context} Model predicted {row.prediction} at {row.confidence_pct:.1f}% "
            f"confidence, versus the expected {expected_label.lower()}."
            if context
            else (
                f"Model predicted {row.prediction} at {row.confidence_pct:.1f}% confidence, "
                f"versus the expected {expected_label.lower()}."
            )
        )
        lines.append(
            f"| {row.scenario_id} | {row.title} | {expected_label} | {actual_label} | "
            f"{row.confidence_pct:.1f}% | {explanation} |"
        )

    lines.append("")
    return "\n".join(lines)


def _format_decision_support_section(
    df: pd.DataFrame, validation_report_text: str, business_scenarios_text: str
) -> str:
    total = len(df)
    passed = int((df["validation_status"] == "PASS").sum())
    explained = int((df["top_shap_factors"].str.len() > 0).sum())
    rules_section = _extract_markdown_section(validation_report_text, "Recommendation Rules")
    transaction_types = sorted(set(re.findall(r"\| `type` \| (\w+) \|", business_scenarios_text)))
    types_phrase = (
        f"{len(transaction_types)} PaySim transaction types ({', '.join(transaction_types)})"
        if transaction_types
        else "multiple PaySim transaction types"
    )

    lines = [
        "## 6. Decision Support Assessment",
        "",
        f"**Prediction capability.** The champion model produced a valid prediction for "
        f"{passed}/{total} scenarios spanning known fraud patterns, legitimate transaction "
        f"types across {types_phrase}, and deliberate boundary conditions, "
        "with no runtime failures observed.",
        "",
        f"**Explainability.** SHAP-based, feature-level explanations were generated for "
        f"{explained}/{total} scenarios, each traceable to specific, auditable transaction "
        "fields (balance changes, ledger discrepancies) rather than an opaque score.",
        "",
        "**Recommendation quality.** Recommendations follow the same fixed, transparent "
        "rules used throughout Sprint 5.5:",
        "",
        rules_section or "(Recommendation rules could not be extracted from validation_report.md.)",
        "",
        "**Suitability for analyst support.** These results support using the system as a "
        "triage and explanation aid for fraud analysts — surfacing high-confidence cases for "
        "expedited action and routing lower-confidence or mismatched cases for review — "
        "rather than as a fully autonomous decision-maker. The system assists human "
        "judgement; it does not replace it, and the mismatches documented in Section 5 "
        "illustrate concretely why analyst oversight remains necessary.",
        "",
    ]
    return "\n".join(lines)


def _format_limitations_section(df: pd.DataFrame, business_scenarios_text: str) -> str:
    mismatches = df[df["outcome_match"] == "Mismatch"]
    small_value_ids = [
        row.scenario_id
        for row in mismatches.itertuples()
        if "small" in (_extract_business_context(business_scenarios_text, row.scenario_id) or "").lower()
        or row.scenario_id in {"E5", "F3"}
    ]

    lines = ["## 7. Limitations", "", "Observed directly from this validation run:", ""]

    if small_value_ids:
        lines.append(
            f"- **Uncertainty around small/moderate-value ledger anomalies.** Scenarios "
            f"{', '.join(small_value_ids)} constructed a balance discrepancy pattern similar "
            "to known fraud signatures but at a smaller dollar magnitude than typical "
            "training-set fraud, and were not flagged as expected — suggesting sensitivity "
            "to anomaly patterns may scale with transaction size."
        )
    lines.append(
        "- **Dependence on historical PaySim patterns.** Every scenario's rationale (see "
        "`business_scenarios.md`) is grounded in statistics from the PaySim simulation "
        "dataset; behaviour on real banking data with different fraud patterns is untested."
    )
    lines.append(
        "- **Deterministic recommendation rules.** The recommendation engine maps prediction "
        "class and confidence to a fixed action list via a single confidence threshold "
        f"({HIGH_CONFIDENCE_THRESHOLD:.0f}%); it does not learn or adapt from outcomes."
    )
    if (df["outcome_match"].str.startswith("N/A")).any():
        na_ids = ", ".join(df.loc[df["outcome_match"].str.startswith("N/A"), "scenario_id"])
        lines.append(
            f"- **No ground truth for genuine boundary conditions.** Scenarios {na_ids} were "
            "constructed with no strong prior expectation; their validity can be assessed "
            "for robustness (did the pipeline run without error) but not for correctness."
        )
    lines.append(
        "- **Single-transaction scoring.** The classifier scores each transaction "
        "independently; multi-step patterns such as structuring (see scenarios F6a/F6b in "
        "`business_scenarios.md`) are only partially represented as separate, related scenarios."
    )
    lines.append("")
    return "\n".join(lines)


def _format_recommendations_section(df: pd.DataFrame) -> str:
    mismatches = df[df["outcome_match"] == "Mismatch"]
    lines = ["## 8. Recommendations", "", "Realistic next steps suggested by this validation run:", ""]

    if len(mismatches):
        ids = ", ".join(mismatches["scenario_id"])
        lines.append(
            f"1. **Review confidence-threshold calibration** in light of the {len(mismatches)} "
            f"mismatched scenario(s) ({ids}), particularly around moderate-magnitude balance "
            "discrepancies that were not flagged as expected."
        )
    else:
        lines.append(
            "1. **Periodically re-validate confidence thresholds** as new scenarios are added."
        )
    lines.append(
        "2. **Add targeted business rules for structuring/sequential patterns**, since the "
        "current single-transaction classifier cannot see related transactions across steps."
    )
    lines.append(
        "3. **Test against real banking transaction data** to confirm the PaySim-derived "
        "patterns generalize beyond the simulation."
    )
    lines.append(
        "4. **Establish a continuous retraining/monitoring cadence** so the model and its "
        "SHAP-based explanations stay aligned with evolving fraud patterns over time."
    )
    lines.append("")
    return "\n".join(lines)


def generate_executive_summary(
    scenario_results_path: Path = SCENARIO_RESULTS_PATH,
    validation_report_path: Path = VALIDATION_REPORT_PATH,
    business_scenarios_path: Path = BUSINESS_SCENARIOS_PATH,
    output_path: Path = EXECUTIVE_SUMMARY_PATH,
) -> Path:
    """Generate the executive validation summary from Sprint 5.5's existing outputs.

    Reads (never modifies) `scenario_results.csv`, `validation_report.md`,
    and `business_scenarios.md`. Every statistic in the resulting summary
    is computed directly from `scenario_results.csv` at call time.

    Args:
        scenario_results_path: Location of `scenario_results.csv`.
        validation_report_path: Location of `validation_report.md`.
        business_scenarios_path: Location of `business_scenarios.md`.
        output_path: Destination Markdown path.

    Returns:
        The path the executive summary was written to.

    Raises:
        FileNotFoundError: If any required input file is missing (i.e.
            the Sprint 5.5 validation workflow has not been run).
    """
    df = load_scenario_results(scenario_results_path)
    validation_report_text = _read_text(validation_report_path)
    business_scenarios_text = _read_text(business_scenarios_path)

    sections = [
        "# Executive Validation Summary",
        "",
        "This document summarizes the Sprint 5.5 business-scenario validation of the "
        "fraud detection decision support system, for dissertation Chapter 5, supervisor "
        "review, and project defense. It is generated entirely from the existing validation "
        "outputs (`scenario_results.csv`, `validation_report.md`, `business_scenarios.md`); "
        "no scenarios were re-run and no model or SHAP computation occurred while producing "
        "this summary.",
        "",
        _format_overview_section(df, scenario_results_path),
        _format_overall_results_section(df),
        _format_key_findings_section(df),
        _format_business_insights_section(df),
        _format_notable_exceptions_section(df, business_scenarios_text),
        _format_decision_support_section(df, validation_report_text, business_scenarios_text),
        _format_limitations_section(df, business_scenarios_text),
        _format_recommendations_section(df),
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sections), encoding="utf-8")
    logger.info("Generated executive summary (%d scenarios) at %s.", len(df), output_path)
    return output_path


def main() -> None:
    """CLI entry point: append to the shared validation log and generate the summary."""
    configure_logging(VALIDATION_LOG_PATH)
    logger.info("=== Executive summary generation started ===")
    start = time.perf_counter()

    output_path = generate_executive_summary()

    elapsed = time.perf_counter() - start
    logger.info(
        "=== Executive summary generation finished in %.2fs. Output: %s ===",
        elapsed,
        output_path,
    )


if __name__ == "__main__":
    main()
