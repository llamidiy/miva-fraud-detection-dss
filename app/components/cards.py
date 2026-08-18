"""Reusable card components: metric cards, info boxes, status badges."""

from typing import Literal, Optional

import streamlit as st

StatusKind = Literal["success", "warning", "danger", "info", "neutral"]


def metric_card(label: str, value: str, delta: Optional[str] = None, help_text: Optional[str] = None) -> None:
    """Render a bordered metric card.

    Pass ``value="Loading..."`` for metrics not yet wired to real data —
    this renders in a distinct muted/italic style via the
    `metric-value--loading` CSS class.

    Args:
        label: Short metric name (e.g. ``"Fraud Flagged"``).
        value: The metric's display value.
        delta: Optional secondary line (e.g. a trend or comparison).
        help_text: Optional caption shown below the value.
    """
    is_loading = value.strip().lower() == "loading..."
    value_class = "metric-value metric-value--loading" if is_loading else "metric-value"
    with st.container(border=True):
        st.markdown(f'<div class="metric-label">{label}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="{value_class}">{value}</div>', unsafe_allow_html=True)
        if delta:
            st.markdown(f'<div class="metric-delta">{delta}</div>', unsafe_allow_html=True)
        if help_text:
            st.caption(help_text)


def info_box(message: str, kind: StatusKind = "info") -> None:
    """Render a styled info/success/warning/danger box.

    Args:
        message: The message to display (HTML is not escaped, so pass
            plain text or trusted markup only).
        kind: Visual style variant.
    """
    st.markdown(f'<div class="info-box info-box--{kind}">{message}</div>', unsafe_allow_html=True)


def status_badge_html(label: str, kind: StatusKind = "neutral") -> str:
    """Build an inline HTML status badge for embedding inside other markdown.

    Args:
        label: The badge text.
        kind: Visual style variant.

    Returns:
        An HTML ``<span>`` snippet. Render it via
        ``st.markdown(..., unsafe_allow_html=True)``.
    """
    return f'<span class="status-badge status-badge--{kind}">{label}</span>'


def status_badge(label: str, kind: StatusKind = "neutral") -> None:
    """Render a standalone status badge.

    Args:
        label: The badge text.
        kind: Visual style variant.
    """
    st.markdown(status_badge_html(label, kind), unsafe_allow_html=True)


def placeholder_card(title: str, description: str = "Loading...") -> None:
    """Render an empty/placeholder card for content not yet wired to the backend.

    Args:
        title: The card's title.
        description: Placeholder body text, typically ``"Loading..."``
            or a short note on what will appear here in Phase 2.
    """
    with st.container(border=True):
        st.markdown(f'<div class="placeholder-card-title">{title}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="placeholder-card-body">{description}</div>', unsafe_allow_html=True)
