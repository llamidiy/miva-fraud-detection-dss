"""End-to-end explainability workflow orchestration and reporting.

Ties together sampling, global feature importance, SHAP analysis, and a
local explanation example into a single Markdown report
(`reports/model_interpretation.md`) suitable as a basis for a
dissertation's model interpretation chapter.
"""

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    EXPLAINABILITY_LOG_PATH,
    EXPLAINABILITY_SAMPLE_SIZE,
    MODEL_INTERPRETATION_REPORT_PATH,
    MODEL_RESULTS_PATH,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET_COLUMN,
)
from src.explainability.explainer import FraudExplainer
from src.explainability.feature_importance import run_feature_importance
from src.explainability.local_explanations import explain_single_transaction
from src.explainability.sampling import create_explainability_sample
from src.explainability.shap_analysis import run_shap_analysis
from src.utils.logger import configure_logging

logger = logging.getLogger(__name__)


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavored Markdown table.

    Avoids a dependency on the optional `tabulate` package that
    `pandas.DataFrame.to_markdown` normally requires.

    Args:
        df: The DataFrame to render.

    Returns:
        The table as a Markdown string.
    """
    def _format_cell(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(_format_cell(value) for value in row) + " |")
    return "\n".join(lines)


def _relative_to_report(path: Path) -> str:
    """Express a `reports/`-relative path for use as a Markdown image link.

    Args:
        path: An absolute path under `REPORTS_DIR`.

    Returns:
        The path relative to `REPORTS_DIR` (where the report itself
        lives), so image links resolve correctly regardless of where the
        project is checked out.
    """
    return path.relative_to(REPORTS_DIR).as_posix()


def _load_model_comparison_table(path: Path = MODEL_RESULTS_PATH) -> pd.DataFrame:
    """Load the existing Sprint 4 model comparison table, if available.

    Args:
        path: Location of `reports/model_results.csv`.

    Returns:
        The comparison table, or an empty DataFrame if it cannot be found
        (the report still generates without it).
    """
    if not path.exists():
        logger.warning("Model comparison table not found at %s; skipping.", path)
        return pd.DataFrame()
    return pd.read_csv(path)


def _top_mean_abs_shap_features(explanation: Any, top_n: int = 5) -> pd.DataFrame:
    """Rank features by mean absolute SHAP value across the sample.

    Args:
        explanation: A multi-row `shap.Explanation`.
        top_n: Number of top features to return.

    Returns:
        A DataFrame with ``feature`` and ``mean_abs_shap`` columns.
    """
    mean_abs = np.abs(explanation.values).mean(axis=0)
    df = pd.DataFrame({"feature": explanation.feature_names, "mean_abs_shap": mean_abs})
    return df.sort_values("mean_abs_shap", ascending=False).head(top_n).reset_index(drop=True)


def _format_champion_model_section(explainer: FraudExplainer, comparison_df: pd.DataFrame) -> str:
    metadata = explainer.artifacts.metadata
    lines = [
        "## Champion Model",
        "",
        f"- **Model:** {metadata['model_name']} (`{metadata['algorithm']}`)",
        f"- **Trained:** {metadata['training_timestamp']}",
        f"- **Target column:** `{metadata['target_column']}`",
        f"- **Random state:** {metadata['random_state']}",
        "",
    ]
    if not comparison_df.empty and "model_name" in comparison_df.columns:
        lines.append(
            "XGBoost was selected as the champion model for explainability because it matched "
            "Random Forest's near-perfect detection quality on the full held-out test set while "
            "training in a fraction of the time, making it the more practical choice for a "
            "production decision support system:"
        )
        lines.append("")
        cols = [
            c
            for c in [
                "model_name",
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "roc_auc",
                "training_time_seconds",
            ]
            if c in comparison_df.columns
        ]
        lines.append(_df_to_markdown(comparison_df[cols].round(4)))
        lines.append("")
    return "\n".join(lines)


def _format_performance_section(explainer: FraudExplainer) -> str:
    metrics = explainer.artifacts.metadata["evaluation_metrics"]
    cm = metrics["confusion_matrix"]
    lines = [
        "## Performance Summary",
        "",
        f"- **Accuracy:** {metrics['accuracy']:.4f}",
        f"- **Precision:** {metrics['precision']:.4f}",
        f"- **Recall:** {metrics['recall']:.4f}",
        f"- **F1-score:** {metrics['f1_score']:.4f}",
        f"- **ROC-AUC:** {metrics['roc_auc']:.4f}" if metrics["roc_auc"] is not None else "",
        "",
        "**Confusion matrix** (rows = actual, columns = predicted):",
        "",
        "|              | Predicted Legit | Predicted Fraud |",
        "|--------------|-----------------|------------------|",
        f"| Actual Legit | {cm[0][0]} | {cm[0][1]} |",
        f"| Actual Fraud | {cm[1][0]} | {cm[1][1]} |",
        "",
    ]
    return "\n".join(line for line in lines if line is not None)


def _format_feature_importance_section(importance_df: pd.DataFrame, png_path: Path) -> str:
    top = importance_df.head(10)
    lines = [
        "## Global Feature Importance",
        "",
        "Native (gain-based) feature importance from the trained XGBoost model — "
        "the top 10 features by contribution to the model's split decisions:",
        "",
        _df_to_markdown(top.round(4)),
        "",
        f"![Feature Importance]({_relative_to_report(png_path)})",
        "",
    ]
    return "\n".join(lines)


def _format_shap_findings_section(
    explanation: Any, figure_paths: dict[str, Path], sample_size: int
) -> str:
    top_shap = _top_mean_abs_shap_features(explanation)
    lines = [
        "## SHAP Findings",
        "",
        f"SHAP values were computed on a stratified representative sample of "
        f"{sample_size:,} transactions (see `data/explainability/shap_sample.csv`), "
        "preserving the full dataset's fraud/non-fraud ratio.",
        "",
        "Top features by mean absolute SHAP value across the sample:",
        "",
        _df_to_markdown(top_shap.round(4)),
        "",
        f"![SHAP Summary]({_relative_to_report(figure_paths['shap_summary'])})",
        "",
        f"![SHAP Beeswarm]({_relative_to_report(figure_paths['shap_beeswarm'])})",
        "",
        f"![SHAP Bar]({_relative_to_report(figure_paths['shap_bar'])})",
        "",
        f"![SHAP Dependence: amount]({_relative_to_report(figure_paths['shap_dependence_amount'])})",
        "",
        f"![SHAP Dependence: balanceDeltaOrig]({_relative_to_report(figure_paths['shap_dependence_balanceDeltaOrig'])})",
        "",
    ]
    return "\n".join(lines)


def _format_local_explanation_section(local_result: dict[str, Any]) -> str:
    lines = [
        "## Local Explanation Example",
        "",
        "A single fraud-labeled transaction from the representative sample, explained end-to-end:",
        "",
        "```",
        local_result["narrative"],
        "```",
        "",
        f"![Waterfall Plot]({_relative_to_report(Path(local_result['waterfall_plot_path']))})",
        "",
    ]
    return "\n".join(lines)


def _format_business_interpretation_section(top_shap: pd.DataFrame) -> str:
    top_feature_names = ", ".join(f"`{f}`" for f in top_shap["feature"].head(3))
    lines = [
        "## Business Interpretation",
        "",
        "The model's decisions are driven primarily by transaction economics rather than "
        f"arbitrary correlations: {top_feature_names} consistently rank among the strongest "
        "signals across the representative sample. This aligns with the exploratory analysis "
        "in Sprint 2, which found fraud concentrated in `TRANSFER`/`CASH_OUT` transactions and "
        "associated with balances that do not reconcile the way legitimate transfers do "
        "(e.g. an origin account emptied to exactly zero, or a destination balance that does "
        "not increase by the transacted amount). The SHAP dependence plots make this concrete: "
        "large positive balance discrepancies and unusually high transaction amounts both push "
        "predictions toward fraud.",
        "",
        "Because the model's reasoning is traceable to specific, auditable transaction fields "
        "rather than opaque interactions, its outputs can be defended to compliance reviewers "
        "and, where required, to affected customers.",
        "",
    ]
    return "\n".join(lines)


def _format_recommendations_section() -> str:
    return "\n".join(
        [
            "## Practical Recommendations",
            "",
            "1. **Use the model as a triage layer, not an autonomous blocker.** Route "
            "high-probability predictions to manual review rather than auto-declining "
            "transactions outright, given the precision/recall trade-offs documented in "
            "Sprint 4.",
            "2. **Prioritize review queues using the top contributing factors**, not just the "
            "raw probability score — investigators can verify balance-discrepancy and "
            "zero-balance flags directly against ledger records.",
            "3. **Retire or recalibrate the existing `isFlaggedFraud` rule engine** in favor of "
            "the model's output; Sprint 3's exploratory analysis showed it catches only a "
            "negligible share of actual fraud.",
            "4. **Monitor SHAP feature rankings over time** as transaction patterns evolve, and "
            "regenerate this report whenever the model is retrained.",
            "",
        ]
    )


def _format_limitations_section(sample_size: int) -> str:
    return "\n".join(
        [
            "## Limitations",
            "",
            "- **Synthetic data.** PaySim is a simulation of mobile money transactions; fraud "
            "patterns and feature relationships may not fully transfer to a real production "
            "dataset.",
            f"- **Explainability sample size.** SHAP analysis was computed on a stratified "
            f"sample of {sample_size:,} transactions rather than the full 6.36M-row dataset, "
            "for computational tractability; global patterns should be broadly representative "
            "but rare interaction effects may not surface.",
            "- **TreeExplainer approximation.** SHAP's TreeExplainer computes exact values for "
            "tree ensembles under a feature-independence assumption; correlated features (e.g. "
            "`amount` and `logAmount`) can split credit between themselves.",
            "- **No temporal validation.** The train/validation/test split is a random "
            "stratified split, not a time-based holdout, so the evaluation does not directly "
            "test performance on future, unseen fraud patterns.",
            "- **Class imbalance handling.** Training relied on SMOTE-based oversampling; this "
            "improves recall but can shift the model's calibration, so predicted probabilities "
            "should be read as relative risk scores rather than well-calibrated probabilities.",
            "",
        ]
    )


def generate_model_interpretation_report(
    explainer: FraudExplainer,
    comparison_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    importance_png_path: Path,
    shap_result: dict[str, Any],
    local_result: dict[str, Any],
    sample_size: int,
    output_path: Path = MODEL_INTERPRETATION_REPORT_PATH,
) -> Path:
    """Assemble and save the full Markdown model interpretation report.

    Args:
        explainer: An initialized `FraudExplainer`.
        comparison_df: The Sprint 4 model comparison table (may be empty).
        importance_df: The global feature importance table.
        importance_png_path: Path to the saved feature importance figure.
        shap_result: The result dict from
            `~src.explainability.shap_analysis.run_shap_analysis`.
        local_result: The result dict from
            `~src.explainability.local_explanations.explain_single_transaction`.
        sample_size: Number of rows in the representative sample, for
            reporting.
        output_path: Destination Markdown path.

    Returns:
        The path the report was written to.
    """
    top_shap = _top_mean_abs_shap_features(shap_result["explanation"])

    sections = [
        "# Model Interpretation Report",
        "",
        "This report documents the explainability analysis of the fraud detection "
        "decision support system's champion model, and is intended as a basis for the "
        "dissertation's model interpretation chapter.",
        "",
        _format_champion_model_section(explainer, comparison_df),
        _format_performance_section(explainer),
        _format_feature_importance_section(importance_df, importance_png_path),
        _format_shap_findings_section(shap_result["explanation"], shap_result["figures"], sample_size),
        _format_local_explanation_section(local_result),
        _format_business_interpretation_section(top_shap),
        _format_recommendations_section(),
        _format_limitations_section(sample_size),
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sections), encoding="utf-8")
    logger.info("Generated model interpretation report at %s.", output_path)
    return output_path


def run_explainability_workflow(
    sample_size: int = EXPLAINABILITY_SAMPLE_SIZE,
    use_cache: bool = True,
) -> Path:
    """Run the full Sprint 5 explainability workflow end-to-end.

    Draws the representative sample, computes global feature importance,
    runs SHAP analysis, generates one local explanation example, and
    writes the final Markdown report. No model is retrained or modified.

    Args:
        sample_size: Number of rows for the representative explainability
            sample.
        use_cache: Whether to reuse cached SHAP values/sample when valid.

    Returns:
        The path to the generated `model_interpretation.md` report.
    """
    logger.info("=== Explainability workflow started ===")
    start = time.perf_counter()

    sample = create_explainability_sample(sample_size=sample_size, use_cache=use_cache)
    explainer = FraudExplainer()

    importance_result = run_feature_importance(explainer)

    shap_result = run_shap_analysis(explainer, sample, use_cache=use_cache)

    fraud_rows = sample[sample[TARGET_COLUMN] == 1]
    example_transaction = fraud_rows.sample(n=1, random_state=RANDOM_STATE).drop(
        columns=[TARGET_COLUMN]
    )
    reference_sample = sample.drop(columns=[TARGET_COLUMN])
    local_result = explain_single_transaction(explainer, example_transaction, reference_sample)

    comparison_df = _load_model_comparison_table()

    report_path = generate_model_interpretation_report(
        explainer=explainer,
        comparison_df=comparison_df,
        importance_df=importance_result["dataframe"],
        importance_png_path=importance_result["png_path"],
        shap_result=shap_result,
        local_result=local_result,
        sample_size=len(sample),
    )

    elapsed = time.perf_counter() - start
    logger.info("=== Explainability workflow finished in %.2fs. Report: %s ===", elapsed, report_path)
    return report_path


def main() -> None:
    """CLI entry point: configure logging and run the full explainability workflow."""
    configure_logging(EXPLAINABILITY_LOG_PATH)
    run_explainability_workflow()


if __name__ == "__main__":
    main()
