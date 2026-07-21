"""Session isolation and deterministic web-app state."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import streamlit as st

from ..web_helpers import (
    cleanup_session_outputs,
    new_session_id,
    parse_zone_yaml,
    zones_to_yaml,
)
from .config import WEB_OUTPUT_ROOT, ZONE_PRESETS


def _default_zone_yaml() -> str:
    path, image_path = next(
        (value for value in ZONE_PRESETS.values() if value[0] is not None),
        (None, None),
    )
    if path is None or not path.is_file() or image_path is None:
        return "zones: []\n"
    image = cv2.imread(str(image_path))
    if image is None:
        return "zones: []\n"
    height, width = image.shape[:2]
    zones = parse_zone_yaml(
        path.read_text(encoding="utf-8"), source_width=width, source_height=height
    )
    return zones_to_yaml(zones)


def initialize_session() -> None:
    """Create an isolated output root and all durable UI defaults."""
    st.session_state.setdefault("web_session_id", new_session_id())
    session_root = WEB_OUTPUT_ROOT / st.session_state.web_session_id
    session_root.mkdir(parents=True, exist_ok=True)
    if not st.session_state.get("retention_checked"):
        cleanup_session_outputs(
            WEB_OUTPUT_ROOT,
            keep_session_id=st.session_state.web_session_id,
            max_age_hours=24.0,
            max_sessions=20,
        )
        st.session_state.retention_checked = True
    defaults: dict[str, Any] = {
        "web_output_root": session_root,
        "ui_language": "en",
        "zone_yaml": _default_zone_yaml(),
        "zone_preset_loaded": next(iter(ZONE_PRESETS)),
        "run_history": [],
        "selected_run": None,
        "monitoring_profile": "balanced",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def record_run(run_dir: Path, source_label: str) -> None:
    """Place the newest successful run first without duplicating history."""
    history = [item for item in st.session_state.run_history if item["path"] != str(run_dir)]
    history.insert(0, {"path": str(run_dir), "source": source_label})
    st.session_state.run_history = history[:20]
    st.session_state.selected_run = str(run_dir)
