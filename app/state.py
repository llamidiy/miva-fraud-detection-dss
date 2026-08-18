"""Shared, cached application state for the Streamlit session.

This module — together with `services/` — is the *only* part of the
presentation layer allowed to import from `src.*`. Pages and components
must go through `services/` (and this module's `get_app_state`)
instead of touching the backend directly.

`get_app_state()` aggregates the champion model's metrics, the
validation summary, and the executive summary into one cached object
(`st.cache_data`), so Streamlit's rerun-on-every-interaction model
doesn't re-read report files constantly.

Note: the cached champion model *explainer* (`FraudExplainer`) is
deliberately defined in `services.explainability_service`, not here —
`explainability_service`/`prediction_service` need it, and this module
depends on those services' leaf siblings (`metrics_service`,
`validation_service`, `report_service`), so keeping the explainer out
of this module avoids a circular import between `state` and `services`.
"""

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_APP_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _APP_DIR.parent
for _path in (_APP_DIR, _PROJECT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import streamlit as st

from src.config import UI_INTEGRATION_LOG_PATH
from src.utils.logger import configure_logging

from services.metrics_service import ModelMetrics, get_model_metrics
from services.report_service import ReportContent, get_executive_summary
from services.validation_service import ValidationSummary, get_validation_summary

# Configured once, at first import of this module (guaranteed to happen
# before any service does real work, since every service and page in
# Phase 2 imports from here). Kept out of `app.py` so the shell itself
# never needs to import `src.*` directly.
configure_logging(UI_INTEGRATION_LOG_PATH)

logger = logging.getLogger(__name__)

SystemStatus = Literal["Operational", "Degraded"]


@dataclass
class AppState:
    """Shared application state for the current Streamlit session.

    Attributes:
        model_metrics: The champion model's metrics summary.
        validation_summary: The Sprint 5.5 business-scenario validation summary.
        executive_summary: The Sprint 5.5.1 executive summary report content.
        system_status: ``"Operational"`` if every summary above loaded
            without error, else ``"Degraded"``.
    """

    model_metrics: ModelMetrics
    validation_summary: ValidationSummary
    executive_summary: ReportContent
    system_status: SystemStatus


@st.cache_data(show_spinner="Loading dashboard data...")
def get_app_state() -> AppState:
    """Load and cache the shared dashboard/report state for this session.

    Returns:
        The populated `AppState`.
    """
    logger.info("Loading dashboard application state...")
    start = time.perf_counter()

    model_metrics = get_model_metrics()
    validation_summary = get_validation_summary()
    executive_summary = get_executive_summary()

    has_error = any([model_metrics.error, validation_summary.error, executive_summary.error])
    status: SystemStatus = "Degraded" if has_error else "Operational"

    elapsed = time.perf_counter() - start
    logger.info("Dashboard application state loaded in %.3fs (status=%s).", elapsed, status)

    return AppState(
        model_metrics=model_metrics,
        validation_summary=validation_summary,
        executive_summary=executive_summary,
        system_status=status,
    )
