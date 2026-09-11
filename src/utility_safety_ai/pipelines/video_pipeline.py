"""End-to-end video inference pipeline."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from ..detection.yolo_detector import YoloDetector
from ..events.event import SafetyEvent
from ..events.summary import write_summary
from ..monitoring import MonitoringProfile, MonitoringQuality
from ..privacy.face_blur import blur_faces
from ..rules.rule_engine import RuleEngine
from ..tracking.simple_tracker import SimpleTracker
from ..utils.paths import OutputPaths
from ..visualization.annotator import annotate_image
from ..zones.zone import Zone
from ._artifacts import (
    RunArtifacts,
    detector_manifest,
    file_source_integrity,
    finalize_video,
    open_video_writer,
    redact_source,
    zones_manifest,
)

logger = logging.getLogger(__name__)


def run_video_pipeline(
    source_path: str | Path,
    output_root: str | Path,
    detector: YoloDetector,
    zones: list[Zone],
    rule_engine: RuleEngine | None = None,
    blur_faces_enabled: bool = True,
    max_frames: int | None = None,
    privacy_mode: str = "gaussian",
    progress_callback: Callable[[int, int | None], None] | None = None,
    *,
    run_id: str | None = None,
    overwrite: bool = False,
    audit_source: str | None = None,
    monitoring_profile: MonitoringProfile | None = None,
    event_sink: Callable[[SafetyEvent], None] | None = None,
) -> list[SafetyEvent]:
    """Run inference on a video and persist an immutable, auditable run."""
    source_path = Path(source_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Video source not found: {source_path}")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be greater than zero")

    # Decode one frame before creating outputs. A corrupt/empty input therefore
    # cannot delete or publish misleading audit artifacts.
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Could not open video: {source_path}")
    first_ok, first_frame = cap.read()
    if not first_ok or first_frame is None or first_frame.size == 0:
        cap.release()
        raise ValueError(f"Video contains no decodable frames: {source_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not 0.1 <= fps <= 240.0:
        fps = 30.0
    height, width = first_frame.shape[:2]
    raw_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    expected_total = raw_frame_count if raw_frame_count > 0 else None
    if max_frames is not None and expected_total is not None:
        expected_total = min(expected_total, max_frames)

    engine = rule_engine or RuleEngine(
        zones=zones,
        rules_config=(monitoring_profile.rule_config() if monitoring_profile else None),
    )
    output_paths = OutputPaths(output_root, run_id=run_id, overwrite=overwrite)
    safe_source = redact_source(
        audit_source if audit_source is not None else source_path,
        portable_local=True,
    )
    output_paths.start_manifest(
        source_type="video",
        source=safe_source,
        model=detector_manifest(detector),
        source_integrity=file_source_integrity(source_path),
        config={
            "privacy_blur_enabled": blur_faces_enabled,
            "privacy_mode": privacy_mode,
            "max_frames": max_frames,
            "input_fps": fps,
            "frame_size": [width, height],
            "zones": zones_manifest(zones),
            "rule_engine": {
                "cooldown_seconds": engine.cooldown_seconds,
                "rules_config": engine.rules_config,
            },
            "monitoring_profile": (
                monitoring_profile.manifest() if monitoring_profile else {"name": "custom"}
            ),
        },
    )
    out_video_path = output_paths.videos / f"{source_path.stem}_annotated.mp4"
    writer: cv2.VideoWriter | None = None
    all_events: list[SafetyEvent] = []
    total_detections = 0
    frame_index = 0

    try:
        writer = open_video_writer(
            out_video_path,
            fps=fps,
            frame_size=(width, height),
        )
        artifacts = RunArtifacts(
            output_paths,
            event_sink=event_sink,
            association_min_score=engine.association_min_score,
            association_ambiguity_margin=engine.association_ambiguity_margin,
        )
        quality = MonitoringQuality()
        engine.reset()
        reset_tracking = getattr(detector, "reset_tracking", None)
        if callable(reset_tracking):
            reset_tracking()
        fallback_tracker = SimpleTracker(
            iou_threshold=(
                monitoring_profile.tracker_iou_threshold if monitoring_profile else 0.22
            ),
            max_age=monitoring_profile.tracker_max_age if monitoring_profile else 12,
        )
        shared_metadata = {
            "run_id": output_paths.run_id,
            "confidence_threshold": getattr(detector, "conf", None),
            "fps": fps,
            "privacy_blur_enabled": blur_faces_enabled,
            "privacy_mode": privacy_mode,
            "frame_size": (width, height),
        }

        frame: np.ndarray | None = first_frame
        while frame is not None and (max_frames is None or frame_index < max_frames):
            frame_started = perf_counter()
            time_seconds = frame_index / fps
            inference_started = perf_counter()
            detections = detector.track(frame)
            inference_seconds = perf_counter() - inference_started
            detections = fallback_tracker.update(detections)

            display_frame = blur_faces(
                frame.copy(),
                detections,
                enabled=blur_faces_enabled,
                mode=privacy_mode,
            )
            evaluation = engine.evaluate_frame(
                detections,
                source_type="video",
                source_path=safe_source,
                frame_index=frame_index,
                time_seconds=time_seconds,
                metadata=shared_metadata,
            )
            annotated = annotate_image(
                display_frame,
                zones,
                detections,
                evaluation.active_findings,
            )
            if annotated.shape[:2] != (height, width):
                raise ValueError(
                    f"Annotated frame size changed from {(width, height)} to {(annotated.shape[1], annotated.shape[0])}"
                )
            writer.write(annotated)

            updated_events = artifacts.persist_frame(
                display_frame,
                detections,
                evaluation.new_events,
                evaluation=evaluation,
                source_type="video",
                source_path=safe_source,
                frame_index=frame_index,
                time_seconds=time_seconds,
                metadata=shared_metadata,
            )
            quality.observe(
                detections,
                evaluation,
                inference_seconds=inference_seconds,
                pipeline_seconds=perf_counter() - frame_started,
                association_min_score=engine.association_min_score,
                association_ambiguity_margin=engine.association_ambiguity_margin,
            )
            all_events.extend(updated_events)
            total_detections += len(detections)
            frame_index += 1
            if progress_callback is not None:
                progress_callback(frame_index, expected_total)

            ok, next_frame = cap.read()
            frame = next_frame if ok and next_frame is not None and next_frame.size else None

        writer.release()
        writer = None
        cap.release()
        if expected_total is not None and frame_index < expected_total:
            raise ValueError(
                f"Video decoding ended early: processed {frame_index} of {expected_total} expected frames"
            )
        video_encoding = finalize_video(out_video_path, expected_frames=frame_index)
        write_summary(all_events, output_paths.events)
        artifacts.persist_lifecycle(engine.end_stream("run_completed"))
        quality.write(output_paths.root)
        output_paths.complete_manifest(
            metrics={
                "frames_processed": frame_index,
                "video_encoding": video_encoding,
                "detections": total_detections,
                "events": len(all_events),
            }
        )
    except BaseException as exc:
        cap.release()
        if writer is not None:
            writer.release()
        try:
            output_paths.fail_manifest(exc)
        except Exception:
            logger.exception("Failed to record failed-run manifest for %s", output_paths.run_id)
        raise

    logger.info(
        "Completed video run %s at %s: %d frame(s), %d event(s), %d detection(s)",
        output_paths.run_id,
        output_paths.published_root,
        frame_index,
        len(all_events),
        total_detections,
    )
    return all_events
