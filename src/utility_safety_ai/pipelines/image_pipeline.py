"""End-to-end image inference pipeline."""

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


def run_image_pipeline(
    source_path: str | Path,
    output_root: str | Path,
    detector: YoloDetector,
    zones: list[Zone],
    rule_engine: RuleEngine | None = None,
    blur_faces_enabled: bool = False,
) -> tuple[np.ndarray, list[SafetyEvent]]:
    """Run inference on a single image and persist outputs.

    Args:
        source_path: Path to the input image.
        output_root: Directory where outputs are written.
        detector: Initialized YOLO detector.
        zones: Restricted zones to check.
        rule_engine: Rule engine instance. If None, a default engine is used.
        blur_faces_enabled: Whether to blur privacy-sensitive regions.

    Returns:
        Annotated image and list of emitted safety events.
    """
    source_path = Path(source_path)
    output_paths = OutputPaths(output_root)
    output_paths.ensure_directories()
    output_paths.reset_logs()

    image = cv2.imread(str(source_path))
    if image is None:
        raise ValueError(f"Could not read image: {source_path}")

    detections = detector.predict(image)
    # Assign stable IDs for single-image analysis so summaries can count persons.
    tracker = SimpleTracker(iou_threshold=0.1)
    detections = tracker.update(detections)

    # Apply privacy blur on a copy so the original frame stays available for
    # internal debugging / audit if ever needed.
    display_image = blur_faces(image.copy(), detections, enabled=blur_faces_enabled)

    engine = rule_engine or RuleEngine(zones=zones)
    shared_metadata = {
        "confidence_threshold": detector.conf,
        "privacy_blur_enabled": blur_faces_enabled,
    }
    events = engine.evaluate(
        detections,
        source_type="image",
        source_path=str(source_path),
        metadata=shared_metadata,
    )

    annotated = annotate_image(display_image, zones, detections, events)
    out_image_path = output_paths.images / source_path.name
    cv2.imwrite(str(out_image_path), annotated)
    logger.info("Saved annotated image to %s", out_image_path)

    event_logger = EventLogger(output_paths.events)
    compliance_reporter = ComplianceReporter(output_paths.events)
    detection_logger = DetectionLogger(output_paths.events)

    compliance_records, _ = associate_ppe_to_persons(detections)
    compliance_reporter.write(compliance_records)

    updated_events: list[SafetyEvent] = []
    for event in events:
        snapshot_path = _save_snapshot(display_image, event, output_paths.snapshots)
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
        updated_events.append(event)

    event_logger.log_all(updated_events)
    write_summary(updated_events, output_paths.events)

    detection_logger.log_all(
        detections,
        source_type="image",
        source_path=str(source_path),
        metadata=shared_metadata,
    )

    logger.info(
        "Event summary: %d event(s), %d zone intrusion(s), %d PPE violation(s), %d detection(s)",
        len(updated_events),
        sum(1 for e in updated_events if e.event_type == "zone_intrusion"),
        sum(1 for e in updated_events if e.event_type.startswith("missing_")),
        len(detections),
    )

    return annotated, updated_events


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
