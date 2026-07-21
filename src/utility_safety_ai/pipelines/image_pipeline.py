"""End-to-end image inference pipeline."""

from __future__ import annotations

import logging
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
    checked_imwrite,
    detector_manifest,
    file_source_integrity,
    redact_source,
    zones_manifest,
)

logger = logging.getLogger(__name__)


def run_image_pipeline(
    source_path: str | Path,
    output_root: str | Path,
    detector: YoloDetector,
    zones: list[Zone],
    rule_engine: RuleEngine | None = None,
    blur_faces_enabled: bool = False,
    privacy_mode: str = "gaussian",
    *,
    run_id: str | None = None,
    overwrite: bool = False,
    audit_source: str | None = None,
    monitoring_profile: MonitoringProfile | None = None,
) -> tuple[np.ndarray, list[SafetyEvent]]:
    """Run image inference and persist an immutable, auditable run.

    Input decoding is validated before any output directory is created. Results
    are stored beneath ``<output_root>/runs/<run_id>``; successful completion
    updates ``<output_root>/latest.json``.
    """
    source_path = Path(source_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Image source not found: {source_path}")
    image = cv2.imread(str(source_path))
    if image is None or image.size == 0:
        raise ValueError(f"Could not decode image: {source_path}")

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
        source_type="image",
        source=safe_source,
        model=detector_manifest(detector),
        source_integrity=file_source_integrity(source_path),
        config={
            "privacy_blur_enabled": blur_faces_enabled,
            "privacy_mode": privacy_mode,
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

    try:
        artifacts = RunArtifacts(
            output_paths,
            association_min_score=engine.association_min_score,
            association_ambiguity_margin=engine.association_ambiguity_margin,
        )
        quality = MonitoringQuality()
        reset_tracking = getattr(detector, "reset_tracking", None)
        if callable(reset_tracking):
            reset_tracking()
        frame_started = perf_counter()
        inference_started = perf_counter()
        detections = detector.predict(image)
        inference_seconds = perf_counter() - inference_started
        # Assign stable IDs so person-level compliance and summaries are useful.
        detections = SimpleTracker(iou_threshold=0.1).update(detections)
        display_image = blur_faces(
            image.copy(), detections, enabled=blur_faces_enabled, mode=privacy_mode
        )

        engine.reset()
        shared_metadata = {
            "run_id": output_paths.run_id,
            "confidence_threshold": getattr(detector, "conf", None),
            "privacy_blur_enabled": blur_faces_enabled,
            "privacy_mode": privacy_mode,
            "frame_size": (image.shape[1], image.shape[0]),
        }
        evaluation = engine.evaluate_frame(
            detections,
            source_type="image",
            source_path=safe_source,
            metadata=shared_metadata,
        )
        annotated = annotate_image(
            display_image,
            zones,
            detections,
            evaluation.active_findings,
        )

        suffix = source_path.suffix if source_path.suffix.lower() in {".jpg", ".jpeg", ".png"} else ".jpg"
        out_image_path = output_paths.images / f"{source_path.stem}_annotated{suffix}"
        checked_imwrite(out_image_path, annotated)

        updated_events = artifacts.persist_frame(
            display_image,
            detections,
            evaluation.new_events,
            source_type="image",
            source_path=safe_source,
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
        write_summary(updated_events, output_paths.events)
        quality.write(output_paths.root)
        output_paths.complete_manifest(
            metrics={
                "frames_processed": 1,
                "detections": len(detections),
                "events": len(updated_events),
            }
        )
    except BaseException as exc:
        try:
            output_paths.fail_manifest(exc)
        except Exception:
            logger.exception("Failed to record failed-run manifest for %s", output_paths.run_id)
        raise

    logger.info(
        "Completed image run %s at %s: %d event(s), %d detection(s)",
        output_paths.run_id,
        output_paths.published_root,
        len(updated_events),
        len(detections),
    )
    return annotated, updated_events
