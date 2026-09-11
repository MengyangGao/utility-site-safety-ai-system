"""Application services that isolate Streamlit from inference pipelines."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..detection.yolo_detector import YoloDetector
from ..monitoring import MonitoringProfile
from ..pipelines.camera_pipeline import run_camera_pipeline
from ..pipelines.image_pipeline import run_image_pipeline
from ..pipelines.video_pipeline import run_video_pipeline
from ..rules.rule_engine import RuleEngine
from ..utils.paths import generate_run_id
from ..web_helpers import NormalizedZone, temporary_upload, zones_to_domain


@dataclass(frozen=True)
class AnalysisSettings:
    """Validated controls for one analysis run."""

    model_path: str
    device: str | None
    confidence: float
    nms_iou: float
    cooldown_seconds: float
    blur_faces: bool
    privacy_mode: str
    profile: MonitoringProfile
    privacy_reason: str = "operator_setting"


@dataclass(frozen=True)
class AnalysisResult:
    """A successful run ready to render in the evidence workspace."""

    run_dir: Path
    source_label: str
    event_count: int


def _detector(settings: AnalysisSettings) -> YoloDetector:
    return YoloDetector(
        model_path=settings.model_path,
        device=settings.device,
        conf=settings.confidence,
        iou=settings.nms_iou,
    )


def _engine(settings: AnalysisSettings, zones: list[NormalizedZone]) -> RuleEngine:
    return RuleEngine(
        zones=zones_to_domain(zones),
        cooldown_seconds=settings.cooldown_seconds,
        rules_config=settings.profile.rule_config(),
    )


def analyse_file(
    upload: Any,
    *,
    is_video: bool,
    output_root: Path,
    zones: list[NormalizedZone],
    settings: AnalysisSettings,
    max_frames: int | None = None,
) -> AnalysisResult:
    """Analyse a Streamlit upload without retaining the raw temporary file."""
    run_id = generate_run_id()
    with temporary_upload(upload) as source:
        detector = _detector(settings)
        engine = _engine(settings, zones)
        if is_video:
            events = run_video_pipeline(
                source,
                output_root,
                detector,
                zones_to_domain(zones),
                rule_engine=engine,
                blur_faces_enabled=settings.blur_faces,
                privacy_mode=settings.privacy_mode,
                max_frames=max_frames,
                run_id=run_id,
                audit_source=upload.name,
                monitoring_profile=settings.profile,
            )
        else:
            _annotated, events = run_image_pipeline(
                source,
                output_root,
                detector,
                zones_to_domain(zones),
                rule_engine=engine,
                blur_faces_enabled=settings.blur_faces,
                privacy_mode=settings.privacy_mode,
                run_id=run_id,
                audit_source=upload.name,
                monitoring_profile=settings.profile,
            )
    return AnalysisResult(output_root / "runs" / run_id, upload.name, len(events))


def analyse_path(
    source: Path,
    *,
    output_root: Path,
    zones: list[NormalizedZone],
    settings: AnalysisSettings,
    source_label: str | None = None,
) -> AnalysisResult:
    """Analyse the sample image included with the repository."""
    run_id = generate_run_id()
    detector = _detector(settings)
    engine = _engine(settings, zones)
    _annotated, events = run_image_pipeline(
        source,
        output_root,
        detector,
        zones_to_domain(zones),
        rule_engine=engine,
        blur_faces_enabled=settings.blur_faces,
        privacy_mode=settings.privacy_mode,
        run_id=run_id,
        audit_source=source.name,
        monitoring_profile=settings.profile,
        privacy_reason=settings.privacy_reason,
    )
    return AnalysisResult(output_root / "runs" / run_id, source_label or source.name, len(events))


def analyse_camera(
    source: str,
    *,
    output_root: Path,
    zones: list[NormalizedZone],
    settings: AnalysisSettings,
    max_frames: int,
    frame_callback: Callable[[np.ndarray, dict], None] | None = None,
) -> AnalysisResult:
    """Capture a bounded camera/RTSP evidence run."""
    run_id = generate_run_id()
    detector = _detector(settings)
    engine = _engine(settings, zones)
    events = run_camera_pipeline(
        source,
        output_root,
        detector,
        zones_to_domain(zones),
        rule_engine=engine,
        blur_faces_enabled=settings.blur_faces,
        privacy_mode=settings.privacy_mode,
        max_frames=max_frames,
        frame_callback=frame_callback,
        run_id=run_id,
        monitoring_profile=settings.profile,
    )
    safe_label = "camera" if source.isdigit() else "network stream"
    return AnalysisResult(output_root / "runs" / run_id, safe_label, len(events))
