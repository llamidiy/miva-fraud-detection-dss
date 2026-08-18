"""Report content service.

Loads Sprint 5.5/5.5.1's existing, frozen Markdown reports — the
executive summary, validation report, and business scenario catalog —
as plain text. No report is regenerated here.
"""

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_APP_DIR = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _APP_DIR.parent
for _path in (_APP_DIR, _PROJECT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import streamlit as st

from src.config import BUSINESS_SCENARIOS_PATH, EXECUTIVE_SUMMARY_PATH, VALIDATION_REPORT_PATH

logger = logging.getLogger(__name__)


@dataclass
class ReportContent:
    """A single loaded report, ready for display.

    Attributes:
        title: Human-readable report title.
        content: The report's raw Markdown text (empty if unavailable).
        available: Whether the report file was found and read successfully.
        error: Set if loading failed.
    """

    title: str
    content: str
    available: bool
    error: Optional[str] = None


def _load_report(title: str, path: Path) -> ReportContent:
    start = time.perf_counter()
    try:
        if not path.exists():
            raise FileNotFoundError(f"Report not found at {path}")
        content = path.read_text(encoding="utf-8")
        logger.info("Loaded report '%s' in %.3fs.", title, time.perf_counter() - start)
        return ReportContent(title=title, content=content, available=True)
    except Exception as exc:
        logger.exception("Failed to load report '%s'.", title)
        return ReportContent(title=title, content="", available=False, error=str(exc))


@st.cache_data(show_spinner="Loading executive summary...")
def get_executive_summary() -> ReportContent:
    """Load the Sprint 5.5.1 executive summary report.

    Returns:
        The report's `ReportContent`.
    """
    return _load_report("Executive Summary", EXECUTIVE_SUMMARY_PATH)


@st.cache_data(show_spinner="Loading validation report...")
def get_validation_report() -> ReportContent:
    """Load the Sprint 5.5 validation report.

    Returns:
        The report's `ReportContent`.
    """
    return _load_report("Validation Report", VALIDATION_REPORT_PATH)


@st.cache_data(show_spinner="Loading business scenarios...")
def get_business_scenarios_report() -> ReportContent:
    """Load the Sprint 5.5 business scenario catalog.

    Returns:
        The report's `ReportContent`.
    """
    return _load_report("Business Scenarios", BUSINESS_SCENARIOS_PATH)
