"""Live camera / webcam / RTSP inference pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from ..compliance.compliance_reporter import ComplianceReporter
from ..compliance.person_ppe_association import associate_ppe_to_persons
from ..detection.yolo_detector import YoloDetector
from ..events.detection_logger import DetectionLogger
from ..events.event import SafetyEvent
from ..events.event_logger import EventLogger
from ..events.summary import write_summary
from ..privacy.face_blur import blur_faces
from ..rules.rule_engine import RuleEngine
from ..tracking.simple_tracker import SimpleTracker
from ..utils.paths import OutputPaths
from ..visualization.annotator import annotate_image
from ..zones.zone import Zone

logger = logging.getLogger(__name__)


def _resolve_capture_source(source: str | int) -> str | int:
    """Convert string webcam indices to integers; keep RTSP/URL strings as-is."""
    if isinstance(source, int):
        return source
    if source.isdigit():
        return int(source)
    return source


def run_camera_pipeline(
    source: str | int,
    output_root: str | Path,
    detector: YoloDetector,
    zones: list[Zone],
    rule_engine: RuleEngine | None = None,
    blur_faces_enabled: bool = False,
    duration_seconds: float | None = None,
    max_frames: int | None = None,
    display: bool = False,
) -> list[SafetyEvent]:
    """Run live inference on a webcam, RTSP stream, or other video capture source.

    Args:
        source: Webcam index (e.g. ``0``), RTSP URL, or local camera path.
        output_root: Directory where outputs are written.
        detector: Initialized YOLO detector.
        zones: Restricted zones to check.
        rule_engine: Rule engine instance. If None, a default engine is used.
        blur_faces_enabled: Whether to blur privacy-sensitive regions.
        duration_seconds: Optional runtime limit in seconds.
        max_frames: Optional frame limit.
        display: Whether to show a live preview window (requires a display).

    Returns:
        List of all emitted safety events.
    """
    output_paths = OutputPaths(output_root)
    output_paths.ensure_directories()
    output_paths.reset_logs()

    capture_source = _resolve_capture_source(source)
    cap = cv2.VideoCapture(capture_source)
    if not cap.isOpened():
        raise ValueError(f"Could not open camera/source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    out_video_path = output_paths.videos / "camera_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

    engine = rule_engine or RuleEngine(zones=zones)
    event_logger = EventLogger(output_paths.events)
    compliance_reporter = ComplianceReporter(output_paths.events)
    fallback_tracker = SimpleTracker()
    all_events: list[SafetyEvent] = []
    total_detections = 0
    frame_index = 0

    logger.info(
        "Started live inference from %s (%dx%d @ %.1f fps). Press 'q' in the preview window to stop.",
        source,
        width,
        height,
        fps,
    )

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Camera stream ended or frame read failed at frame %d", frame_index)
                break

            time_seconds = frame_index / fps
            if duration_seconds is not None and time_seconds >= duration_seconds:
                break
            if max_frames is not None and frame_index >= max_frames:
                break

            detections = detector.track(frame)
            if any(d.track_id is None for d in detections):
                detections = fallback_tracker.update(detections)

            if blur_faces_enabled:
                frame = blur_faces(frame, detections, enabled=True)

            events = engine.evaluate(
                detections,
                source_type="camera",
                source_path=str(source),
                frame_index=frame_index,
                time_seconds=time_seconds,
                metadata={"confidence_threshold": detector.conf, "fps": fps},
            )

            annotated = annotate_image(frame, zones, detections, events)
            writer.write(annotated)

            updated_frame_events: list[SafetyEvent] = []
            for event in events:
                snapshot_path = _save_snapshot(frame, event, output_paths.snapshots)
                if snapshot_path:
                    event = SafetyEvent(
                        event_id=event.event_id,
                        timestamp=event.timestamp,
                        source_type=event.source_type,
                        source_path=event.source_path,
                        frame_index=event.frame_index,
                        time_seconds=event.time_seconds,
                        risk_level=event.risk_level,
                        event_type=event.event_type,
                        description=event.description,
                        person_track_id=event.person_track_id,
                        bbox=event.bbox,
                        zone_id=event.zone_id,
                        zone_name=event.zone_name,
                        snapshot_path=str(snapshot_path),
                        metadata=event.metadata,
                    )
                updated_frame_events.append(event)

            compliance_records, _ = associate_ppe_to_persons(detections)
            compliance_reporter.write(
                compliance_records,
                frame_index=frame_index,
                time_seconds=time_seconds,
            )

            all_events.extend(updated_frame_events)
            event_logger.log_all(updated_frame_events)
            DetectionLogger(output_paths.events).log_all(
                detections,
                source_type="camera",
                source_path=str(source),
                frame_index=frame_index,
                time_seconds=time_seconds,
            )
            total_detections += len(detections)

            if display:
                cv2.imshow("Utility Site Safety AI", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_index += 1
    finally:
        cap.release()
        writer.release()
        if display:
            cv2.destroyAllWindows()

    write_summary(all_events, output_paths.events)
    logger.info(
        "Saved annotated camera clip to %s. Event summary: %d event(s), %d zone intrusion(s), %d PPE violation(s), %d total detection(s)",
        out_video_path,
        len(all_events),
        sum(1 for e in all_events if e.event_type == "zone_intrusion"),
        sum(1 for e in all_events if e.event_type.startswith("missing_")),
        total_detections,
    )
    return all_events


def _save_snapshot(
    image: np.ndarray,
    event: SafetyEvent,
    snapshots_dir: Path,
) -> Path | None:
    if event.bbox is None:
        return None
    x1, y1, x2, y2 = map(int, event.bbox)
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2, x1:x2]
    snapshot_path = snapshots_dir / f"{event.event_id}.jpg"
    cv2.imwrite(str(snapshot_path), crop)
    return snapshot_path
