"""Logging setup helpers."""

from __future__ import annotations

import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    """Keep diagnostics on stderr so JSON/CSV stdout remains machine-readable."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root = logging.getLogger("utility_safety_ai")
    root.setLevel(level)
    if not root.handlers:
        root.addHandler(handler)
