"""Validation summary service.

Loads Sprint 5.5's existing, frozen business-scenario validation results
(`reports/validation/scenario_results.csv`) — no scenario is re-run
here.
"""

import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

_APP_DIR = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _APP_DIR.parent
for _path in (_APP_DIR, _PROJECT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pandas as pd
import streamlit as st

from src.config import SCENARIO_RESULTS_PATH

logger = logging.getLogger(__name__)


@dataclass
class ValidationSummary:
    """UI-friendly summary of Sprint 5.5's business-scenario validation run.

    Attributes:
        total_scenarios: Number of scenarios executed.
        passed: Number that completed with `validation_status == "PASS"`.
        failed: Number that completed with `validation_status == "FAIL"`.
        matches: Number where the actual recommendation matched the
            expected outcome.
        mismatches: Number where it did not.
        exploratory: Number with no strong prior expectation (`N/A`).
        scenarios: The full per-scenario result rows.
        error: Set if the summary could not be loaded; other fields
            default to zero/empty in that case.
    """

    total_scenarios: int
    passed: int
    failed: int
    matches: int
    mismatches: int
    exploratory: int
    scenarios: list[dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


@st.cache_data(show_spinner="Loading validation results...")
def get_validation_summary() -> ValidationSummary:
    """Load the Sprint 5.5 scenario validation summary.

    Returns:
        A `ValidationSummary`. On failure, `error` is set and all counts
        default to zero.
    """
    start = time.perf_counter()
    try:
        if not SCENARIO_RESULTS_PATH.exists():
            raise FileNotFoundError(f"Scenario results not found at {SCENARIO_RESULTS_PATH}")

        df = pd.read_csv(SCENARIO_RESULTS_PATH)
        outcome_match = df["outcome_match"].astype(str)

        summary = ValidationSummary(
            total_scenarios=len(df),
            passed=int((df["validation_status"] == "PASS").sum()),
            failed=int((df["validation_status"] == "FAIL").sum()),
            matches=int((outcome_match == "Match").sum()),
            mismatches=int((outcome_match == "Mismatch").sum()),
            exploratory=int(outcome_match.str.startswith("N/A").sum()),
            scenarios=df.to_dict("records"),
        )
        logger.info("Loaded validation summary in %.3fs.", time.perf_counter() - start)
        return summary

    except Exception as exc:
        logger.exception("Failed to load validation summary.")
        return ValidationSummary(
            total_scenarios=0,
            passed=0,
            failed=0,
            matches=0,
            mismatches=0,
            exploratory=0,
            scenarios=[],
            error=str(exc),
        )
