"""Persist every detection to JSONL and CSV for audit and reporting."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..utils.paths import OUTPUT_SCHEMA_VERSION
from .event import Detection

DETECTION_FIELDS = [
    "schema_version",
    "timestamp",
    "source_type",
    "source_path",
    "frame_index",
    "time_seconds",
    "class_id",
    "class_name",
    "confidence",
    "bbox",
    "track_id",
    "metadata",
]


class DetectionLogger:
    """Append every detection to ``detections.jsonl`` and ``detections.csv``."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.jsonl_path = self.output_dir / "detections.jsonl"
        self.csv_path = self.output_dir / "detections.csv"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path.touch(exist_ok=True)
        self._csv_header_written = self.csv_path.exists() and self.csv_path.stat().st_size > 0
        if not self._csv_header_written:
            with self.csv_path.open("w", encoding="utf-8", newline="") as file:
                csv.DictWriter(file, fieldnames=DETECTION_FIELDS).writeheader()
            self._csv_header_written = True

    def log(
        self,
        detection: Detection,
        source_type: str,
        source_path: str,
        frame_index: int | None = None,
        time_seconds: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Write a single detection to both JSONL and CSV."""
        record = {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_type": source_type,
            "source_path": source_path,
            "frame_index": frame_index,
            "time_seconds": time_seconds,
            **asdict(detection),
            "metadata": metadata or {},
        }
        self._append_jsonl(record)
        self._append_csv(record)

    def log_all(
        self,
        detections: list[Detection],
        source_type: str,
        source_path: str,
        frame_index: int | None = None,
        time_seconds: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        for detection in detections:
            self.log(detection, source_type, source_path, frame_index, time_seconds, metadata)

    def _append_jsonl(self, record: dict[str, Any]) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def _append_csv(self, record: dict[str, Any]) -> None:
        with self.csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=DETECTION_FIELDS)
            writer.writerow({name: self._serialize(record.get(name)) for name in DETECTION_FIELDS})

    @staticmethod
    def _serialize(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (list, dict, tuple)):
            return json.dumps(value)
        return str(value)
