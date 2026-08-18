"""Markdown text helpers for displaying existing report content.

Pure string/regex parsing over Markdown text already returned by
`services.report_service` — no `src.*` import, no report generation or
recomputation. Used so pages can reuse a specific section of an
existing report (e.g. "Limitations") without re-implementing report
generation.
"""

import re
from typing import Optional


def extract_markdown_section(text: str, heading: str) -> Optional[str]:
    """Extract the body of a top-level (`##`) Markdown section by heading text.

    Args:
        text: The full Markdown document.
        heading: The exact heading text to find (without ``##``).

    Returns:
        The section's body text, or ``None`` if the heading is not found.
    """
    pattern = rf"^## {re.escape(heading)}\s*\n(.*?)(?=\n## |\Z)"
    match = re.search(pattern, text, flags=re.DOTALL | re.MULTILINE)
    return match.group(1).strip() if match else None
