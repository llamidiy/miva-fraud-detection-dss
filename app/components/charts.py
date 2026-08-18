"""Chart placeholder components.

Phase 1 renders empty, clearly-labeled placeholder areas rather than
mock/fake chart data, so nobody mistakes placeholder output for a real
result. Phase 2 will replace `chart_placeholder` calls with real
Plotly/Matplotlib figures backed by `src.explainability` outputs.
"""

from typing import Literal

import streamlit as st

ChartSize = Literal["sm", "md", "lg"]


def chart_placeholder(label: str = "Chart will appear here", size: ChartSize = "md") -> None:
    """Render an empty, dashed-border placeholder where a chart will later appear.

    Args:
        label: Text shown inside the placeholder area.
        size: Placeholder height variant (``"sm"``, ``"md"``, or
            ``"lg"``) — maps to a fixed-height CSS class rather than an
            inline style.
    """
    st.markdown(
        f"""
        <div class="chart-placeholder chart-placeholder--{size}">
            <div class="chart-placeholder__icon">📊</div>
            <div class="chart-placeholder__label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
