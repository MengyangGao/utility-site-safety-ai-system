"""Shared audit-artifact handling for image, video, and camera pipelines."""

from __future__ import annotations

import csv
import hashlib
import os
import re
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import cv2
import numpy as np

from ..compliance.compliance_reporter import ComplianceReporter
from ..compliance.person_ppe_association import associate_ppe_to_persons
from ..events.detection_logger import DetectionLogger
from ..events.event import Detection, SafetyEvent
from ..events.event_logger import EventLogger
from ..utils.paths import OutputPaths
from ..zones.zone import Zone

_SENSITIVE_QUERY_KEYS = re.compile(
    r"(?:access[_-]?token|api[_-]?key|auth|credential|key|password|secret|signature|token)",
    re.IGNORECASE,
)
_PPE_CLASS_NAMES = {
    "helmet",
    "vest",
    "gloves",
    "boots",
    "goggles",
    "no_helmet",
    "no_vest",
    "no_gloves",
    "no_boots",
    "no_goggles",
    "no_goggle",
}


def redact_source(
    source: str | Path | int,
    *,
    portable_local: bool = False,
) -> str:
    """Remove URL secrets and optionally hide absolute local path prefixes."""
    value = str(source)
    try:
        parts = urlsplit(value)
    except ValueError:
        return value
    if not parts.scheme or not parts.netloc:
        path = Path(value)
        return path.name if portable_local and path.is_absolute() else value

    hostname = parts.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = hostname
    try:
        if parts.port is not None:
            netloc = f"{netloc}:{parts.port}"
    except ValueError:
        netloc = hostname

    query = urlencode(
        [
            (key, "REDACTED" if _SENSITIVE_QUERY_KEYS.search(key) else value)
            for key, value in parse_qsl(parts.query, keep_blank_values=True)
        ]
    )
    return urlunsplit((parts.scheme, netloc, parts.path, query, ""))


def file_source_integrity(path: str | Path) -> dict[str, Any]:
    """Return stable file evidence without recording its absolute location."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "source_sha256": digest.hexdigest(),
        "size_bytes": path.stat().st_size,
        "integrity_unavailable_reason": None,
    }


def unavailable_source_integrity(reason: str) -> dict[str, Any]:
    """Describe why a live source has no stable whole-input digest."""
    return {
        "source_sha256": None,
        "size_bytes": None,
        "integrity_unavailable_reason": reason,
    }


def detector_manifest(detector: Any) -> dict[str, Any]:
    """Return a JSON-safe description without coupling to one detector backend."""
    model = getattr(detector, "model", None)
    model_path = getattr(model, "ckpt_path", None)
    if model_path is None:
        model_path = getattr(detector, "model_path", None)
    raw_names = getattr(model, "names", {})
    if isinstance(raw_names, dict):
        classes = [str(raw_names[key]).lower() for key in sorted(raw_names)]
    elif isinstance(raw_names, (list, tuple)):
        classes = [str(value).lower() for value in raw_names]
    else:
        classes = []
    ppe_classes = sorted(set(classes) & _PPE_CLASS_NAMES)
    capabilities: list[str] = []
    if "person" in classes:
        capabilities.extend(["person_detection", "zone_intrusion"])
    if ppe_classes:
        capabilities.extend(["ppe_detection", "ppe_rule_events"])
    model_file = Path(model_path) if model_path is not None else None
    model_hash = None
    display_model_path = str(model_path) if model_path is not None else None
    if model_file is not None and model_file.is_file():
        digest = hashlib.sha256()
        with model_file.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        model_hash = digest.hexdigest()
        try:
            display_model_path = str(model_file.resolve().relative_to(Path.cwd().resolve()))
        except ValueError:
            # A basename plus a content hash identifies external weights without
            # publishing a user's home-directory or mounted-volume path.
            display_model_path = model_file.name
    return {
        "detector_class": f"{type(detector).__module__}.{type(detector).__name__}",
        "model_path": display_model_path,
        "model_origin": "local" if model_file is not None and model_file.is_file() else "hub_or_runtime",
        "model_kind": "ppe" if ppe_classes else "general" if classes else "unknown",
        "model_sha256": model_hash,
        "classes": classes,
        "ppe_classes": ppe_classes,
        "capabilities": capabilities,
        "device": str(getattr(detector, "device", "unknown")),
        "confidence_threshold": getattr(detector, "conf", None),
        "iou_threshold": getattr(detector, "iou", None),
    }


def zones_manifest(zones: list[Zone]) -> list[dict[str, Any]]:
    """Serialize the exact zone policy used by a run."""
    return [asdict(zone) for zone in zones]


def checked_imwrite(path: Path, image: np.ndarray) -> Path:
    """Write an image and raise if OpenCV reports a codec/filesystem failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"OpenCV failed to write image: {path}")
    if not path.is_file() or path.stat().st_size == 0:
        raise OSError(f"Image output is missing or empty: {path}")
    return path


def open_video_writer(
    path: Path,
    *,
    fps: float,
    frame_size: tuple[int, int],
) -> cv2.VideoWriter:
    """Create a checked MP4 writer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Some OpenCV type stubs omit this runtime API even though it is present in
    # every supported wheel.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(path), fourcc, fps, frame_size)
    if not writer.isOpened():
        writer.release()
        raise OSError(
            f"Could not initialize MP4 writer for {path} at {frame_size[0]}x{frame_size[1]} @ {fps:.3f} fps"
        )
    return writer


def validate_video_output(path: Path, *, expected_frames: int | None = None) -> None:
    """Fail when a writer appeared to succeed but produced no usable artifact."""
    if not path.is_file() or path.stat().st_size == 0:
        raise OSError(f"Annotated video output is missing or empty: {path}")
    probe = cv2.VideoCapture(str(path))
    try:
        if not probe.isOpened():
            raise OSError(f"Annotated video cannot be reopened: {path}")
        reported_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, frame = probe.read()
        if not ok or frame is None or frame.size == 0:
            raise OSError(f"Annotated video contains no decodable frame: {path}")
        if (
            expected_frames is not None
            and reported_frames > 0
            and reported_frames < expected_frames
        ):
            raise OSError(
                f"Annotated video is incomplete: expected {expected_frames} frame(s), encoded {reported_frames}"
            )
    finally:
        probe.release()


def opencv_display_available() -> bool:
    """Return whether this OpenCV build and host can open a GUI window."""
    if not hasattr(cv2, "imshow"):
        return False
    build_info = cv2.getBuildInformation()
    gui_lines = [line.strip().lower() for line in build_info.splitlines() if "gui:" in line.lower()]
    if any("none" in line for line in gui_lines):
        return False
    has_host_display = os.name == "nt" or sys.platform == "darwin" or bool(
        os.environ.get("DISPLAY")
        or os.environ.get("WAYLAND_DISPLAY")
    )
    return has_host_display


class RunArtifacts:
    """Persist frame-level audit evidence using a single shared implementation."""

    def __init__(self, output_paths: OutputPaths) -> None:
        self.paths = output_paths
        self.event_logger = EventLogger(output_paths.events)
        self.detection_logger = DetectionLogger(output_paths.events)
        self._initialize_compliance_files(output_paths.events)
        self.compliance_reporter = ComplianceReporter(output_paths.events)

    def persist_frame(
        self,
        image: np.ndarray,
        detections: list[Detection],
        events: list[SafetyEvent],
        *,
        source_type: str,
        source_path: str,
        frame_index: int | None = None,
        time_seconds: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[SafetyEvent]:
        """Save snapshots, events, detections, and changed compliance state."""
        updated_events = [self._with_snapshot(image, event) for event in events]
        compliance_records, _ = associate_ppe_to_persons(detections)
        self.compliance_reporter.write(
            compliance_records,
            frame_index=frame_index,
            time_seconds=time_seconds,
        )
        self.event_logger.log_all(updated_events)
        self.detection_logger.log_all(
            detections,
            source_type=source_type,
            source_path=source_path,
            frame_index=frame_index,
            time_seconds=time_seconds,
            metadata=metadata,
        )
        return updated_events

    def _with_snapshot(self, image: np.ndarray, event: SafetyEvent) -> SafetyEvent:
        crop = image
        if event.bbox is not None:
            x1, y1, x2, y2 = map(int, event.bbox)
            height, width = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 > x1 and y2 > y1:
                crop = image[y1:y2, x1:x2]
        snapshot_path = checked_imwrite(
            self.paths.snapshots / f"{event.event_id}.jpg",
            crop,
        )
        return replace(
            event,
            snapshot_path=str(snapshot_path.relative_to(self.paths.root)),
        )

    @staticmethod
    def _initialize_compliance_files(events_dir: Path) -> None:
        """Create schema-valid empty compliance logs before processing frames."""
        events_dir.mkdir(parents=True, exist_ok=True)
        (events_dir / "compliance.jsonl").touch(exist_ok=True)
        csv_path = events_dir / "compliance.csv"
        if not csv_path.exists() or csv_path.stat().st_size == 0:
            with csv_path.open("w", encoding="utf-8", newline="") as file:
                csv.DictWriter(
                    file,
                    fieldnames=ComplianceReporter._fieldnames(),  # noqa: SLF001
                ).writeheader()
