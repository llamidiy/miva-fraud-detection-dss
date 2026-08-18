"""Reusable table components."""

from typing import Any

import pandas as pd
import streamlit as st


def placeholder_table(
    columns: list[str], caption: str = "No data yet — this table will populate in Phase 2."
) -> None:
    """Render an empty table showing only the expected column headers.

    Args:
        columns: Column names to display.
        caption: Caption shown beneath the table.
    """
    st.dataframe(pd.DataFrame(columns=columns), use_container_width=True, hide_index=True)
    st.caption(caption)


def data_table(rows: list[dict[str, Any]], caption: str = "") -> None:
    """Render a table of row dicts, or a friendly empty state if there are none.

    Args:
        rows: Row dicts to display. If empty, shows an info message
            instead of a blank table.
        caption: Optional caption shown beneath the table.
    """
    if not rows:
        st.info("No data available yet.")
        return
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    if caption:
        st.caption(caption)
