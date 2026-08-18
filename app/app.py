"""Streamlit application shell for the Fraud Detection Decision Support System.

Configures global page settings, loads the shared stylesheet, and wires
up sidebar navigation across all seven pages. This phase (Sprint 6 —
Phase 1) builds the presentation shell only: no backend, model, SHAP,
or validation logic is imported or called here or in any page yet.
"""

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

ASSETS_DIR = APP_DIR / "assets"
PAGES_DIR = APP_DIR / "pages"

from components.navigation import render_sidebar_branding  # noqa: E402


def _configure_page() -> None:
    """Set global Streamlit page configuration. Must run before any other st call."""
    favicon_path = ASSETS_DIR / "favicon.png"
    st.set_page_config(
        page_title="Fraud Detection Decision Support System",
        page_icon=str(favicon_path) if favicon_path.exists() else "🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def _load_global_css() -> None:
    """Inject the single shared stylesheet, once, for every page.

    No page or component uses inline `style=` attributes — every visual
    rule lives in `assets/styles.css`.
    """
    css_path = ASSETS_DIR / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _build_navigation() -> "st.navigation":
    """Define the sidebar page order, titles, and icons.

    Returns:
        The configured `st.navigation` object, ready to `.run()`.
    """
    pages = [
        st.Page(PAGES_DIR / "1_Dashboard.py", title="Dashboard", icon="🏠", default=True),
        st.Page(PAGES_DIR / "2_Single_Transaction.py", title="Single Transaction", icon="🔍"),
        st.Page(PAGES_DIR / "3_Batch_Assessment.py", title="Batch Assessment", icon="📂"),
        st.Page(PAGES_DIR / "4_Explainability.py", title="Explainability", icon="🧠"),
        st.Page(PAGES_DIR / "5_Validation.py", title="Validation", icon="✅"),
        st.Page(PAGES_DIR / "6_Model_Performance.py", title="Model Performance", icon="📊"),
        st.Page(PAGES_DIR / "7_About.py", title="About", icon="ℹ️"),
    ]
    return st.navigation(pages)


def main() -> None:
    """Configure and run the application shell."""
    _configure_page()
    _load_global_css()
    render_sidebar_branding()
    navigation = _build_navigation()
    navigation.run()


if __name__ == "__main__":
    main()
