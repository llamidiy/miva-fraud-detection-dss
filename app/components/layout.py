"""Shared page-layout building blocks: titles, section headers, footer."""

import streamlit as st


def page_title(title: str, subtitle: str = "", phase_label: str = "") -> None:
    """Render a consistent page title block.

    Args:
        title: The page's main title.
        subtitle: Optional one-line description shown under the title.
        phase_label: Optional small badge (e.g. ``"Phase 1 — Preview"``)
            shown above the title.
    """
    if phase_label:
        st.markdown(f'<div class="phase-badge">{phase_label}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)
    st.divider()


def section_header(title: str, description: str = "") -> None:
    """Render a consistent section header within a page.

    Args:
        title: The section's title.
        description: Optional short caption shown under the title.
    """
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)
    if description:
        st.caption(description)


def footer() -> None:
    """Render the shared footer present on every page."""
    st.markdown(
        """
        <div class="app-footer">
            <span>MIVA Open University</span>
            <span class="divider">|</span>
            <span>Master of Information Technology</span>
            <span class="divider">|</span>
            <span>Machine Learning-Based Fraud Detection Decision Support System</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
