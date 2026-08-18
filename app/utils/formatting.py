"""Pure formatting helpers for currency, percentages, and text display.

No backend or I/O dependency — safe to use in Phase 1's placeholder
pages and reused unchanged once Phase 2 wires in real data.
"""

from typing import Union

Number = Union[int, float]


def format_currency(value: Number, decimals: int = 2) -> str:
    """Format a numeric amount as a USD currency string.

    Args:
        value: The amount to format.
        decimals: Number of decimal places to show.

    Returns:
        A string like ``"$1,234.50"``.
    """
    return f"${value:,.{decimals}f}"


def format_percentage(value: Number, decimals: int = 1) -> str:
    """Format a numeric value already on a 0-100 scale as a percentage string.

    Args:
        value: The percentage value (e.g. ``99.8`` for 99.8%).
        decimals: Number of decimal places to show.

    Returns:
        A string like ``"99.8%"``.
    """
    return f"{value:.{decimals}f}%"


def format_confidence(confidence_pct: Number) -> str:
    """Format a model confidence score (0-100 scale) for display.

    Args:
        confidence_pct: Confidence as a percentage.

    Returns:
        A string like ``"99.8% confidence"``.
    """
    return f"{confidence_pct:.1f}% confidence"


def format_count(value: int) -> str:
    """Format an integer count with thousands separators.

    Args:
        value: The count to format.

    Returns:
        A string like ``"1,234"``.
    """
    return f"{value:,}"


def truncate_text(text: str, max_length: int = 80) -> str:
    """Truncate long text with an ellipsis for compact display.

    Args:
        text: The text to truncate.
        max_length: Maximum length before truncation, including the
            ellipsis character.

    Returns:
        The original text if short enough, otherwise a truncated copy
        ending in "…".
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - 1].rstrip() + "…"


def title_case_label(raw_label: str) -> str:
    """Convert a snake_case or lower-case identifier into a display label.

    Args:
        raw_label: An identifier like ``"fraud_high_confidence"``.

    Returns:
        A display-friendly label like ``"Fraud High Confidence"``.
    """
    return raw_label.replace("_", " ").strip().title()
