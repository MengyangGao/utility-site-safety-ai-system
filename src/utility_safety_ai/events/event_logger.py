"""Persist safety events to JSONL and CSV."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..utils.paths import OUTPUT_SCHEMA_VERSION
from .event import SafetyEvent

FIELD_NAMES = [
    "schema_version",
    "event_id",
    "timestamp",
    "source_type",
    "source_path",
    "frame_index",
    "time_seconds",
    "risk_level",
    "event_type",
    "description",
    "person_track_id",
    "bbox",
    "zone_id",
    "zone_name",
    "snapshot_path",
    "metadata",
]


class EventLogger:
    """Append safety events to ``events.jsonl`` and ``events.csv``."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.jsonl_path = self.output_dir / "events.jsonl"
        self.csv_path = self.output_dir / "events.csv"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path.touch(exist_ok=True)
        self._csv_header_written = self.csv_path.exists() and self.csv_path.stat().st_size > 0
        if not self._csv_header_written:
            with self.csv_path.open("w", encoding="utf-8", newline="") as file:
                csv.DictWriter(file, fieldnames=FIELD_NAMES).writeheader()
            self._csv_header_written = True

    def log(self, event: SafetyEvent) -> None:
        """Write a single event to both JSONL and CSV."""
        record = {"schema_version": OUTPUT_SCHEMA_VERSION, **asdict(event)}
        self._append_jsonl(record)
        self._append_csv(record)

    def log_all(self, events: list[SafetyEvent]) -> None:
        for event in events:
            self.log(event)

    def _append_jsonl(self, record: dict[str, Any]) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def _append_csv(self, record: dict[str, Any]) -> None:
        with self.csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELD_NAMES)
            writer.writerow({name: self._serialize(record.get(name)) for name in FIELD_NAMES})

    @staticmethod
    def _serialize(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (list, dict, tuple)):
            return json.dumps(value)
        return str(value)
