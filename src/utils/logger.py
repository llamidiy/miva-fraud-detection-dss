"""Shared logging configuration.

Provides a single reusable function for wiring up console + append-mode
file logging, following the same pattern established by the Sprint 3
preprocessing pipeline (see
:func:`src.preprocessing.pipeline._configure_logging`), so any entry
point in the project can log consistently to both the terminal and a
dedicated log file under ``reports/logs/``.
"""

import logging
from pathlib import Path

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def configure_logging(log_path: Path, level: int = logging.INFO) -> None:
    """Configure root logging to write to both the console and a log file.

    The log file is appended to rather than overwritten, so repeated runs
    accumulate a full history. Both handlers share the same log format.
    Safe to call more than once per process: each call replaces any
    previously configured root handlers rather than stacking duplicates.

    Args:
        log_path: Destination log file. Its parent directory is created
            automatically if it does not already exist.
        level: Logging level applied to the root logger and both handlers.
            Defaults to ``logging.INFO``.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(_LOG_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)

    logging.basicConfig(level=level, handlers=[console_handler, file_handler], force=True)
