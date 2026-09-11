"""Locate bundled demo assets in a checkout or an installed wheel."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def resource_root() -> Path:
    override = os.getenv("UTILITY_SAFETY_REPO_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    checkout = Path(__file__).resolve().parents[2]
    if (checkout / "models" / "registry.json").is_file():
        return checkout
    packaged = Path(sys.prefix) / "share" / "utility-safety-ai"
    return packaged if packaged.is_dir() else Path.cwd()
