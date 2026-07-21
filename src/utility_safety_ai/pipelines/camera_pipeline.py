"""Live camera, webcam, and RTSP inference pipeline."""

from __future__ import annotations

import logging
import time
from pathlib import Path

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
    open_video_writer,
    opencv_display_available,
    redact_source,
    unavailable_source_integrity,
    validate_video_output,
    zones_manifest,
)

logger = logging.getLogger(__name__)


def _resolve_capture_source(source: str | int) -> str | int:
    """Convert string webcam indices to integers; keep URLs and paths as-is."""
    if isinstance(source, int):
        return source
    return int(source) if source.isdigit() else source


def _is_network_source(source: str | int) -> bool:
    return isinstance(source, str) and source.lower().startswith(("rtsp://", "rtsps://", "http://", "https://"))


def _reconnect_network_capture(
    source: str,
    *,
    attempts: int,
    delay_seconds: float,
):
    """Try to reconnect a network stream and return ``(capture, first_frame)``."""
    safe_source = redact_source(source)
    for attempt in range(1, attempts + 1):
        logger.warning(
            "Frame read failed for %s; reconnecting (%d/%d)",
            safe_source,
            attempt,
            attempts,
        )
        if delay_seconds:
            time.sleep(delay_seconds)
        capture = cv2.VideoCapture(source)
        ok, frame = capture.read() if capture.isOpened() else (False, None)
        if ok and frame is not None and frame.size:
            return capture, frame
        capture.release()
    return None, None


def run_camera_pipeline(
    source: str | int,
    output_root: str | Path,
    detector: YoloDetector,
    zones: list[Zone],
    rule_engine: RuleEngine | None = None,
    blur_faces_enabled: bool = False,
    privacy_mode: str = "gaussian",
    duration_seconds: float | None = None,
    max_frames: int | None = None,
    display: bool = False,
    *,
    run_id: str | None = None,
    overwrite: bool = False,
    reconnect_attempts: int = 3,
    reconnect_delay_seconds: float = 0.5,
    audit_source: str | None = None,
    monitoring_profile: MonitoringProfile | None = None,
) -> list[SafetyEvent]:
    """Run live inference with monotonic timing and credential-safe logging."""
    if duration_seconds is not None and duration_seconds <= 0:
        raise ValueError("duration_seconds must be greater than zero")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be greater than zero")
    if reconnect_attempts < 0:
        raise ValueError("reconnect_attempts must be non-negative")
    if reconnect_delay_seconds < 0:
        raise ValueError("reconnect_delay_seconds must be non-negative")
    if display and not opencv_display_available():
        raise RuntimeError(
            "OpenCV preview is unavailable in this headless environment. Omit --display and inspect the saved MP4."
        )

    capture_source = _resolve_capture_source(source)
    safe_source = redact_source(
        audit_source if audit_source is not None else source,
        portable_local=True,
    )
    cap = cv2.VideoCapture(capture_source)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Could not open camera/source: {safe_source}")
    first_ok, first_frame = cap.read()
    if not first_ok or first_frame is None or first_frame.size == 0:
        cap.release()
        raise ValueError(f"Camera/source produced no decodable frame: {safe_source}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not 0.1 <= fps <= 240.0:
        fps = 30.0
    height, width = first_frame.shape[:2]

    engine = rule_engine or RuleEngine(
        zones=zones,
        rules_config=(monitoring_profile.rule_config() if monitoring_profile else None),
    )
    output_paths = OutputPaths(output_root, run_id=run_id, overwrite=overwrite)
    output_paths.start_manifest(
        source_type="camera",
        source=safe_source,
        model=detector_manifest(detector),
        source_integrity=unavailable_source_integrity(
            "live camera or network stream has no stable whole-input file"
        ),
        config={
            "privacy_blur_enabled": blur_faces_enabled,
            "privacy_mode": privacy_mode,
            "duration_seconds": duration_seconds,
            "max_frames": max_frames,
            "capture_fps": fps,
            "frame_size": [width, height],
            "reconnect_attempts": reconnect_attempts,
            "reconnect_delay_seconds": reconnect_delay_seconds,
            "display_enabled": display,
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
    out_video_path = output_paths.videos / "camera_annotated.mp4"
    writer: cv2.VideoWriter | None = None
    all_events: list[SafetyEvent] = []
    total_detections = 0
    frame_index = 0
    reconnects = 0

    try:
        writer = open_video_writer(
            out_video_path,
            fps=fps,
            frame_size=(width, height),
        )
        artifacts = RunArtifacts(
            output_paths,
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
            "capture_fps": fps,
            "privacy_blur_enabled": blur_faces_enabled,
            "privacy_mode": privacy_mode,
            "frame_size": (width, height),
        }
        started_at = time.monotonic()
        frame: np.ndarray | None = first_frame

        while frame is not None:
            frame_started = time.perf_counter()
            elapsed = time.monotonic() - started_at
            if duration_seconds is not None and elapsed >= duration_seconds:
                break
            if max_frames is not None and frame_index >= max_frames:
                break

            inference_started = time.perf_counter()
            detections = detector.track(frame)
            inference_seconds = time.perf_counter() - inference_started
            detections = fallback_tracker.update(detections)
            display_frame = blur_faces(
                frame.copy(),
                detections,
                enabled=blur_faces_enabled,
                mode=privacy_mode,
            )
            evaluation = engine.evaluate_frame(
                detections,
                source_type="camera",
                source_path=safe_source,
                frame_index=frame_index,
                time_seconds=elapsed,
                metadata=shared_metadata,
            )
            annotated = annotate_image(
                display_frame,
                zones,
                detections,
                evaluation.active_findings,
            )
            if annotated.shape[:2] != (height, width):
                raise ValueError("Annotated camera frame dimensions changed during processing")
            writer.write(annotated)
            updated_events = artifacts.persist_frame(
                display_frame,
                detections,
                evaluation.new_events,
                source_type="camera",
                source_path=safe_source,
                frame_index=frame_index,
                time_seconds=elapsed,
                metadata=shared_metadata,
            )
            quality.observe(
                detections,
                evaluation,
                inference_seconds=inference_seconds,
                pipeline_seconds=time.perf_counter() - frame_started,
                association_min_score=engine.association_min_score,
                association_ambiguity_margin=engine.association_ambiguity_margin,
            )
            all_events.extend(updated_events)
            total_detections += len(detections)

            if display:
                cv2.imshow("Utility Site Safety AI", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    frame_index += 1
                    break

            frame_index += 1
            if max_frames is not None and frame_index >= max_frames:
                break

            ok, next_frame = cap.read()
            if ok and next_frame is not None and next_frame.size:
                frame = next_frame
                continue

            if _is_network_source(capture_source) and reconnect_attempts:
                cap.release()
                replacement, next_frame = _reconnect_network_capture(
                    str(capture_source),
                    attempts=reconnect_attempts,
                    delay_seconds=reconnect_delay_seconds,
                )
                if replacement is not None:
                    cap = replacement
                    frame = next_frame
                    reconnects += 1
                    # A reconnect is a new continuity segment: never imply that
                    # temporary person IDs remain valid across a stream gap.
                    if callable(reset_tracking):
                        reset_tracking()
                    fallback_tracker.reset()
                    engine.reset()
                    shared_metadata["stream_segment"] = reconnects
                    continue
            logger.warning("Camera stream ended or frame read failed after frame %d", frame_index)
            frame = None

        writer.release()
        writer = None
        cap.release()
        if display:
            cv2.destroyAllWindows()
        validate_video_output(out_video_path, expected_frames=frame_index)
        write_summary(all_events, output_paths.events)
        quality.write(output_paths.root)
        output_paths.complete_manifest(
            metrics={
                "frames_processed": frame_index,
                "detections": total_detections,
                "events": len(all_events),
                "reconnections": reconnects,
                "elapsed_seconds": time.monotonic() - started_at,
            }
        )
    except BaseException as exc:
        cap.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()
        try:
            output_paths.fail_manifest(exc)
        except Exception:
            logger.exception("Failed to record failed-run manifest for %s", output_paths.run_id)
        raise

    logger.info(
        "Completed camera run %s at %s: %d frame(s), %d event(s), %d detection(s)",
        output_paths.run_id,
        output_paths.published_root,
        frame_index,
        len(all_events),
        total_detections,
    )
    return all_events
