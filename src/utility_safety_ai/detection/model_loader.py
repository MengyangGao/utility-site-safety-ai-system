"""Resolve YOLO model weights paths, downloading defaults when necessary."""

from __future__ import annotations

import logging
from pathlib import Path

from ..resources import resource_root

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "yolo11n.pt"
PPE_MODEL_CANDIDATES = [
    "models/ppe_yolo11n.pt",
    "models/ppe_yolo11s.pt",
]
DEFAULT_MODEL_LOCAL_CANDIDATES = ["models/yolo11n.pt"]


def _candidate_locations(candidate: str) -> tuple[Path, ...]:
    """Return deterministic locations for repository-owned model weights."""
    project_root = resource_root()
    cwd_path = Path.cwd() / candidate
    project_path = project_root / candidate
    return (cwd_path,) if cwd_path == project_path else (cwd_path, project_path)


def resolve_model_path(model_path: str | Path | None) -> str | Path:
    """Return a usable model path.

    If ``model_path`` is None, the function first looks for a trained PPE model
    in the project (``models/ppe_yolo11s.pt`` or similar). If no PPE model is
    found, it falls back to ``yolo11n.pt``, which Ultralytics will download
    automatically on first use.

    Args:
        model_path: User-supplied model path or name, or None.

    Returns:
        A path string or Path object that can be passed to ``YOLO()``.
    """
    if model_path is None:
        for candidate in PPE_MODEL_CANDIDATES:
            for location in _candidate_locations(candidate):
                if location.is_file():
                    logger.info("No model specified; using discovered PPE model %s", location)
                    return location
        for candidate in DEFAULT_MODEL_LOCAL_CANDIDATES:
            for location in _candidate_locations(candidate):
                if location.is_file():
                    logger.info(
                        "No PPE model found; using local general detector %s",
                        location,
                    )
                    return location
        logger.info("No model specified; using default model %s", DEFAULT_MODEL)
        return DEFAULT_MODEL

    supplied = str(model_path)
    local_path = Path(model_path).expanduser()
    if local_path.is_file():
        return local_path
    if local_path.exists():
        raise ValueError(f"Model path is not a file: {local_path}")

    # Only bare model names are eligible for Ultralytics auto-download. A missing
    # explicit path should fail here with an actionable message instead of being
    # misinterpreted later as a hub identifier.
    if not local_path.is_absolute() and local_path.parent == Path("."):
        logger.info(
            "Model file not found locally; treating %s as an Ultralytics model name",
            supplied,
        )
        return supplied
    raise FileNotFoundError(f"Model weights file not found: {local_path}")
