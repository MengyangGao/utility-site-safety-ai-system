"""Persist every detection to JSONL and CSV for audit and reporting."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .event import Detection

logger = logging.getLogger(__name__)

DETECTION_FIELDS = [
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
]


class DetectionLogger:
    """Append every detection to ``detections.jsonl`` and ``detections.csv``."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.jsonl_path = self.output_dir / "detections.jsonl"
        self.csv_path = self.output_dir / "detections.csv"
        self._csv_header_written = self.csv_path.exists() and self.csv_path.stat().st_size > 0

    def log(
        self,
        detection: Detection,
        source_type: str,
        source_path: str,
        frame_index: int | None = None,
        time_seconds: float | None = None,
    ) -> None:
        """Write a single detection to both JSONL and CSV."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_type": source_type,
            "source_path": source_path,
            "frame_index": frame_index,
            "time_seconds": time_seconds,
            **asdict(detection),
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
    ) -> None:
        for detection in detections:
            self.log(detection, source_type, source_path, frame_index, time_seconds)

    def _append_jsonl(self, record: dict[str, Any]) -> None:
        try:
            with self.jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to write detection JSONL: %s", exc)

    def _append_csv(self, record: dict[str, Any]) -> None:
        try:
            with self.csv_path.open("a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=DETECTION_FIELDS)
                if not self._csv_header_written:
                    writer.writeheader()
                    self._csv_header_written = True
                writer.writerow({name: self._serialize(record.get(name)) for name in DETECTION_FIELDS})
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to write detection CSV: %s", exc)

    @staticmethod
    def _serialize(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (list, dict, tuple)):
            return json.dumps(value)
        return str(value)
