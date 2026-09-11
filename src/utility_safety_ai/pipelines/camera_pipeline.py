"""Live camera inference with bounded capture and reviewable continuity segments."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from math import isfinite
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
    finalize_video,
    open_video_writer,
    opencv_display_available,
    redact_source,
    unavailable_source_integrity,
    zones_manifest,
)
from .capture import CaptureError, CaptureOptions, FrameCapture

logger = logging.getLogger(__name__)


def run_camera_pipeline(
    source: str | int,
    output_root: str | Path,
    detector: YoloDetector,
    zones: list[Zone],
    rule_engine: RuleEngine | None = None,
    blur_faces_enabled: bool = True,
    privacy_mode: str = "gaussian",
    duration_seconds: float | None = 300,
    max_frames: int | None = None,
    display: bool = False,
    *,
    run_id: str | None = None,
    overwrite: bool = False,
    reconnect_attempts: int = 3,
    reconnect_delay_seconds: float = 0.5,
    audit_source: str | None = None,
    monitoring_profile: MonitoringProfile | None = None,
    event_sink: Callable[[SafetyEvent], None] | None = None,
    capture_options: CaptureOptions | None = None,
    stop_event: threading.Event | None = None,
    frame_callback: Callable[[np.ndarray, dict], None] | None = None,
) -> list[SafetyEvent]:
    """Process a bounded evidence run; network outages never count as success.

    Source time is monotonic host capture time. The annotated MP4 contains processed
    frames at the nominal input FPS and can be time-compressed after dropped frames.
    The timestamp log preserves actual per-frame timing for review.
    """
    if duration_seconds is not None and (
        isinstance(duration_seconds, bool)
        or not isfinite(duration_seconds)
        or duration_seconds <= 0
    ):
        raise ValueError("duration_seconds must be finite and greater than zero")
    if max_frames is not None and (
        isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames <= 0
    ):
        raise ValueError("max_frames must be greater than zero")
    if display and not opencv_display_available():
        raise RuntimeError("OpenCV preview is unavailable; omit --display and inspect saved media.")
    options = capture_options or CaptureOptions(
        reconnect_attempts=reconnect_attempts, reconnect_delay_seconds=reconnect_delay_seconds
    )
    started_at = time.monotonic()
    deadline = started_at + duration_seconds if duration_seconds is not None else None
    capture = FrameCapture(source, options, stop_event=stop_event, deadline=deadline)
    safe_source = redact_source(
        audit_source if audit_source is not None else source, portable_local=True
    )
    engine = rule_engine or RuleEngine(
        zones=zones, rules_config=monitoring_profile.rule_config() if monitoring_profile else None
    )
    output_paths: OutputPaths | None = None
    writer = None
    artifacts: RunArtifacts | None = None
    all_events: list[SafetyEvent] = []
    total_detections = frame_index = 0
    quality = MonitoringQuality()
    try:
        with capture:
            packet = capture.read()
            if packet is None:
                raise CaptureError(f"Camera/source produced no decodable frame: {safe_source}")
            height, width = packet.image.shape[:2]
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
                    "capture_fps": packet.fps,
                    "frame_size": [width, height],
                    "capture": capture.summary(),
                    "display_enabled": display,
                    "recording_timing": "processed frames at nominal input FPS; use events/frame_timestamps.jsonl for host timing",
                    "zones": zones_manifest(zones),
                    "rule_engine": {
                        "cooldown_seconds": engine.cooldown_seconds,
                        "rules_config": engine.rules_config,
                    },
                    "monitoring_profile": monitoring_profile.manifest()
                    if monitoring_profile
                    else {"name": "custom"},
                },
            )
            out_video_path = output_paths.videos / "camera_annotated.mp4"
            writer = open_video_writer(out_video_path, fps=packet.fps, frame_size=(width, height))
            artifacts = RunArtifacts(
                output_paths,
                association_min_score=engine.association_min_score,
                association_ambiguity_margin=engine.association_ambiguity_margin,
                event_sink=event_sink,
            )
            engine.reset()
            reset_tracking = getattr(detector, "reset_tracking", None)
            if callable(reset_tracking):
                reset_tracking()
            fallback_tracker = SimpleTracker(
                iou_threshold=monitoring_profile.tracker_iou_threshold
                if monitoring_profile
                else 0.22,
                max_age=monitoring_profile.tracker_max_age if monitoring_profile else 12,
            )
            segment = packet.segment
            import json

            with (output_paths.events / "frame_timestamps.jsonl").open(
                "w", encoding="utf-8"
            ) as timestamps:
                while packet is not None:
                    frame_started = time.perf_counter()
                    if packet.segment != segment:
                        artifacts.persist_lifecycle(engine.end_stream("stream_gap"))
                        if callable(reset_tracking):
                            reset_tracking()
                        fallback_tracker.reset()
                        segment = packet.segment
                    if packet.image.shape[:2] != (height, width):
                        raise CaptureError(
                            "Source resolution changed; start a new run and recheck the zone policy."
                        )
                    elapsed = max(0.0, packet.captured_at - started_at)
                    age = max(0.0, time.monotonic() - packet.captured_at)
                    metadata = {
                        "run_id": output_paths.run_id,
                        "stream_segment": segment,
                        "confidence_threshold": getattr(detector, "conf", None),
                        "capture_fps": packet.fps,
                        "capture_sequence": packet.sequence,
                        "frame_age_seconds": age,
                        "privacy_blur_enabled": blur_faces_enabled,
                        "privacy_mode": privacy_mode,
                        "frame_size": (width, height),
                    }
                    inference_started = time.perf_counter()
                    detections = fallback_tracker.update(detector.track(packet.image))
                    inference_seconds = time.perf_counter() - inference_started
                    display_frame = blur_faces(
                        packet.image.copy(),
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
                        metadata=metadata,
                    )
                    annotated = annotate_image(
                        display_frame, zones, detections, evaluation.active_findings
                    )
                    writer.write(annotated)
                    events = artifacts.persist_frame(
                        display_frame,
                        detections,
                        evaluation.new_events,
                        source_type="camera",
                        source_path=safe_source,
                        frame_index=frame_index,
                        time_seconds=elapsed,
                        metadata=metadata,
                        evaluation=evaluation,
                    )
                    all_events.extend(events)
                    total_detections += len(detections)
                    timestamps.write(
                        json.dumps(
                            {
                                "frame_index": frame_index,
                                "capture_sequence": packet.sequence,
                                "stream_segment": segment,
                                "capture_elapsed_seconds": elapsed,
                                "queue_age_seconds": age,
                            }
                        )
                        + "\n"
                    )
                    timestamps.flush()
                    quality.observe(
                        detections,
                        evaluation,
                        inference_seconds=inference_seconds,
                        pipeline_seconds=time.perf_counter() - frame_started,
                        association_min_score=engine.association_min_score,
                        association_ambiguity_margin=engine.association_ambiguity_margin,
                        stream_segment=segment,
                    )
                    frame_index += 1
                    if frame_callback:
                        frame_callback(
                            annotated,
                            {
                                "frames": frame_index,
                                "events": len(all_events),
                                "stream_segment": segment,
                                "frames_dropped": capture.frames_dropped,
                                "queue_age_seconds": round(age, 3),
                            },
                        )
                    if display:
                        cv2.imshow("Utility Site Safety AI", annotated)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            capture.stop_reason = "operator_stop"
                            break
                    if max_frames is not None and frame_index >= max_frames:
                        capture.stop_reason = "frame_limit"
                        break
                    packet = capture.read()
            writer.release()
            writer = None
        artifacts.persist_lifecycle(engine.end_stream(capture.stop_reason))
        video_encoding = finalize_video(out_video_path, expected_frames=frame_index)
        write_summary(all_events, output_paths.events)
        quality.write(output_paths.root)
        output_paths.complete_manifest(
            metrics={
                "frames_processed": frame_index,
                "video_encoding": video_encoding,
                "detections": total_detections,
                "events": len(all_events),
                "reconnections": capture.reconnections,
                "elapsed_seconds": time.monotonic() - started_at,
                "capture": capture.summary(),
            }
        )
    except BaseException as exc:
        if writer is not None:
            writer.release()
        if output_paths is not None:
            try:
                if artifacts is not None:
                    artifacts.persist_lifecycle(engine.end_stream("run_failed"))
                write_summary(all_events, output_paths.events)
                quality.write(output_paths.root)
                output_paths.fail_manifest(exc)
            except Exception:
                logger.exception("Failed to record camera diagnostic manifest")
        raise
    finally:
        if display:
            cv2.destroyAllWindows()
    logger.info(
        "Completed camera run %s: %d frames, %d events, %d dropped frames",
        output_paths.run_id,
        frame_index,
        len(all_events),
        capture.frames_dropped,
    )
    return all_events
