"""Static asset path helpers.

Resolves paths to pre-generated report/figure files on disk using only
`pathlib` arithmetic relative to this file's location — no `src.*`
import, since these are static asset locations on a known, documented
directory layout, not backend logic. This is deliberately separate from
`app/services/` (frozen this sprint): it does not compute, load, or
transform any data, it only locates already-existing files by their
stable, documented naming convention.
"""

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = _PROJECT_ROOT / "reports"
VALIDATION_SCREENSHOTS_DIR = REPORTS_DIR / "validation" / "screenshots"
DIAGRAMS_DIR = Path(__file__).resolve().parents[1] / "assets" / "diagrams"


def get_scenario_waterfall_path(scenario_id: str) -> Path:
    """Resolve the pre-generated waterfall plot path for a validation scenario.

    Sprint 5.5 generates one waterfall PNG per scenario at
    `reports/validation/screenshots/waterfall_<scenario_id>.png`. This
    reconstructs that path from its stable, documented naming
    convention rather than adding a new function to the frozen service
    layer this sprint.

    Args:
        scenario_id: A validation scenario ID (e.g. ``"F1"``).

    Returns:
        The expected path to that scenario's waterfall plot. Callers
        should check `.exists()` before use.
    """
    return VALIDATION_SCREENSHOTS_DIR / f"waterfall_{scenario_id}.png"


def get_diagram_path(filename: str) -> Path:
    """Resolve a path under `app/assets/diagrams/`.

    Args:
        filename: The diagram's filename (e.g. ``"system_architecture.png"``).

    Returns:
        The full path. Callers should check `.exists()` before use.
    """
    return DIAGRAMS_DIR / filename
