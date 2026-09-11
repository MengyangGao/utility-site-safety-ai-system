"""Load the optional model runtime with process-local privacy settings."""

from __future__ import annotations

import os


def _disable_usage_events() -> None:
    from ultralytics.utils.events import events

    # Do not call settings.update(): it would change the user's persistent global
    # Ultralytics settings for unrelated projects. This affects only this process.
    events.enabled = False


def load_yolo(model, *, allow_download: bool = False):
    # Set before importing Ultralytics so local inference also skips its online
    # probe. Explicit model/dataset download commands may opt into networking.
    os.environ.setdefault("YOLO_OFFLINE", "0" if allow_download else "1")
    from ultralytics import YOLO

    _disable_usage_events()
    return YOLO(model)
