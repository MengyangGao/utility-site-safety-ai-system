"""Resolve YOLO model weights paths, downloading defaults when necessary."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "yolo11n.pt"
PPE_MODEL_CANDIDATES = [
    "models/ppe_yolo11n.pt",
    "models/ppe_yolo11s.pt",
]


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
            if Path(candidate).exists():
                logger.info("No model specified; using discovered PPE model %s", candidate)
                return candidate
        logger.info("No model specified; using default model %s", DEFAULT_MODEL)
        return DEFAULT_MODEL

    model_path = Path(model_path)
    if model_path.exists():
        return model_path

    # If the user passed a bare filename like "yolov8n.pt", let Ultralytics
    # resolve/download it from its model hub.
    logger.info("Model file not found locally; assuming Ultralytics hub name: %s", model_path)
    return str(model_path)
