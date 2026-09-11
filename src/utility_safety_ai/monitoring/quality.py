"""Run-level quality diagnostics for monitoring and operator review."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from ..compliance.person_ppe_association import (
    NEGATIVE_PPE,
    POSITIVE_PPE,
    associate_ppe_to_persons,
)
from ..events.event import Detection
from ..rules.rule_engine import RuleEvaluation


class MonitoringQuality:
    """Accumulate transparent health indicators without claiming model accuracy."""

    def __init__(self) -> None:
        self.frames = 0
        self.detections = 0
        self.person_observations = 0
        self.tracked_person_observations = 0
        self.ppe_observations = 0
        self.unassociated_ppe_observations = 0
        self.provisional_finding_observations = 0
        self.confirmed_finding_observations = 0
        self.emitted_events = 0
        self.inference_seconds = 0.0
        self.pipeline_seconds = 0.0
        self.unique_track_ids: set[tuple[int, int]] = set()
        self._started_at = perf_counter()

    def observe(
        self,
        detections: list[Detection],
        evaluation: RuleEvaluation,
        *,
        inference_seconds: float = 0.0,
        pipeline_seconds: float = 0.0,
        association_min_score: float = 0.28,
        association_ambiguity_margin: float = 0.08,
        stream_segment: int = 0,
    ) -> None:
        """Record one processed frame and its rule-engine outcome."""
        self.frames += 1
        self.detections += len(detections)
        persons = [item for item in detections if item.class_name == "person"]
        self.person_observations += len(persons)
        tracked = [item for item in persons if item.track_id is not None]
        self.tracked_person_observations += len(tracked)
        self.unique_track_ids.update(
            (stream_segment, item.track_id) for item in tracked if item.track_id is not None
        )
        ppe = [
            item
            for item in detections
            if item.class_name in POSITIVE_PPE or item.class_name in NEGATIVE_PPE
        ]
        _records, unassociated = associate_ppe_to_persons(
            detections,
            iou_threshold=association_min_score,
            ambiguity_margin=association_ambiguity_margin,
        )
        self.ppe_observations += len(ppe)
        self.unassociated_ppe_observations += len(unassociated)
        self.provisional_finding_observations += len(evaluation.provisional_findings)
        self.confirmed_finding_observations += len(evaluation.confirmed_findings)
        self.emitted_events += len(evaluation.new_events)
        self.inference_seconds += max(0.0, float(inference_seconds))
        self.pipeline_seconds += max(0.0, float(pipeline_seconds))

    @staticmethod
    def _ratio(numerator: int, denominator: int) -> float | None:
        return round(numerator / denominator, 4) if denominator else None

    def summary(self) -> dict[str, Any]:
        """Return honest operational indicators suitable for a manifest or UI."""
        wall_seconds = max(0.0, perf_counter() - self._started_at)
        processed_seconds = self.pipeline_seconds or wall_seconds
        possible_findings = (
            self.provisional_finding_observations + self.confirmed_finding_observations
        )
        return {
            "schema_version": "1.0",
            "frames_processed": self.frames,
            "detections": self.detections,
            "person_observations": self.person_observations,
            "tracked_person_observations": self.tracked_person_observations,
            "tracked_person_rate": self._ratio(
                self.tracked_person_observations, self.person_observations
            ),
            "unique_track_ids": len(self.unique_track_ids),
            "ppe_observations": self.ppe_observations,
            "unassociated_ppe_observations": self.unassociated_ppe_observations,
            "ppe_assignment_rate": (
                round(1.0 - self.unassociated_ppe_observations / self.ppe_observations, 4)
                if self.ppe_observations
                else None
            ),
            "provisional_finding_observations": self.provisional_finding_observations,
            "confirmed_finding_observations": self.confirmed_finding_observations,
            "temporal_filter_rate": self._ratio(
                self.provisional_finding_observations, possible_findings
            ),
            "emitted_events": self.emitted_events,
            "inference_seconds": round(self.inference_seconds, 4),
            "processing_seconds": round(processed_seconds, 4),
            "effective_fps": round(self.frames / processed_seconds, 2)
            if processed_seconds > 0
            else None,
            "interpretation": {
                "tracked_person_rate": "continuity coverage, not identity accuracy",
                "ppe_assignment_rate": "geometry assignment coverage, not PPE correctness",
                "temporal_filter_rate": "share of active observations still awaiting confirmation",
            },
        }

    def write(self, directory: str | Path) -> tuple[Path, Path]:
        """Write JSON and one-row CSV diagnostics into a run directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        summary = self.summary()
        json_path = directory / "quality.json"
        csv_path = directory / "quality.csv"
        json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        flat = {key: value for key, value in summary.items() if not isinstance(value, dict)}
        with csv_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(flat))
            writer.writeheader()
            writer.writerow(flat)
        return json_path, csv_path
