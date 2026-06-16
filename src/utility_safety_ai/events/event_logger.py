"""Persist safety events to JSONL and CSV."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .event import SafetyEvent

logger = logging.getLogger(__name__)

FIELD_NAMES = [
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
        self._csv_header_written = self.csv_path.exists() and self.csv_path.stat().st_size > 0

    def log(self, event: SafetyEvent) -> None:
        """Write a single event to both JSONL and CSV."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        record = asdict(event)
        self._append_jsonl(record)
        self._append_csv(record)

    def log_all(self, events: list[SafetyEvent]) -> None:
        for event in events:
            self.log(event)

    def _append_jsonl(self, record: dict[str, Any]) -> None:
        try:
            with self.jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to write JSONL event: %s", exc)

    def _append_csv(self, record: dict[str, Any]) -> None:
        try:
            with self.csv_path.open("a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELD_NAMES)
                if not self._csv_header_written:
                    writer.writeheader()
                    self._csv_header_written = True
                writer.writerow({name: self._serialize(record.get(name)) for name in FIELD_NAMES})
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to write CSV event: %s", exc)

    @staticmethod
    def _serialize(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (list, dict, tuple)):
            return json.dumps(value)
        return str(value)
