"""Persist per-person PPE compliance reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .person_ppe_association import PersonCompliance


class ComplianceReporter:
    """Write person-level PPE compliance reports to JSONL and CSV."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.output_dir / "compliance.jsonl"
        self.csv_path = self.output_dir / "compliance.csv"
        self._csv_header_written = self.csv_path.exists() and self.csv_path.stat().st_size > 0

    def write(
        self,
        records: list[PersonCompliance],
        frame_index: int | None = None,
        time_seconds: float | None = None,
    ) -> tuple[Path, Path]:
        """Append compliance records to JSONL and CSV.

        Args:
            records: Per-person compliance records for a single frame.
            frame_index: Optional frame number (video/camera).
            time_seconds: Optional elapsed time (video/camera).

        Returns:
            Paths to the written JSONL and CSV files.
        """
        rows = [self._to_row(r, frame_index, time_seconds) for r in records]

        with self.jsonl_path.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, default=str) + "\n")

        with self.csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames())
            if not self._csv_header_written:
                writer.writeheader()
                self._csv_header_written = True
            for row in rows:
                writer.writerow(row)

        return self.jsonl_path, self.csv_path

    @staticmethod
    def _to_row(
        record: PersonCompliance,
        frame_index: int | None,
        time_seconds: float | None,
    ) -> dict:
        bbox = record.bbox
        return {
            "person_track_id": record.person_track_id,
            "frame_index": frame_index,
            "time_seconds": time_seconds,
            "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
            "helmet": record.items.get("helmet", "unknown"),
            "vest": record.items.get("vest", "unknown"),
            "gloves": record.items.get("gloves", "unknown"),
            "boots": record.items.get("boots", "unknown"),
            "goggles": record.items.get("goggles", "unknown"),
            "violations": ";".join(record.violations()),
            "positive_classes": ",".join(d.class_name for d in record.positive_detections),
            "negative_classes": ",".join(d.class_name for d in record.negative_detections),
        }

    @staticmethod
    def _fieldnames() -> list[str]:
        return [
            "person_track_id",
            "frame_index",
            "time_seconds",
            "bbox",
            "helmet",
            "vest",
            "gloves",
            "boots",
            "goggles",
            "violations",
            "positive_classes",
            "negative_classes",
        ]
