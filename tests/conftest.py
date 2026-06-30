"""pytest configuration that lets the suite run before the package is installed."""

from __future__ import annotations

import sys
from pathlib import Path


def pytest_configure(config) -> None:  # noqa: ARG001
    """Add ``src/`` to ``sys.path`` so local imports resolve during testing."""
    repo_root = Path(__file__).resolve().parent.parent
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
