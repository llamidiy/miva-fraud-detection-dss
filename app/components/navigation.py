"""Sidebar branding.

Streamlit's built-in `st.navigation` (wired up in `app.py`) renders the
actual page links; this module only adds the logo/name/tagline shown
above them.
"""

from pathlib import Path

import streamlit as st

_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


def render_sidebar_branding() -> None:
    """Render the logo, app name, and tagline at the top of the sidebar."""
    logo_path = _ASSETS_DIR / "logo.png"
    with st.sidebar:
        if logo_path.exists():
            st.image(str(logo_path), use_container_width=True)
        st.markdown('<div class="sidebar-app-name">Fraud Detection DSS</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sidebar-app-tagline">Decision Support System</div>', unsafe_allow_html=True
        )
        st.divider()
