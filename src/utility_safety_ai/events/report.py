"""Convert event JSONL files to other report formats."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def export_report(events_path: str | Path, output_path: str | Path, fmt: str = "csv") -> Path:
    """Export ``events.jsonl`` to ``csv`` (or copy to ``jsonl``).

    Args:
        events_path: Source JSONL events file.
        output_path: Destination file or directory.
        fmt: Target format; currently only ``csv`` and ``jsonl`` are supported.

    Returns:
        Path to the written report file.
    """
    events_path = Path(events_path)
    output_path = Path(output_path)

    if not events_path.exists():
        raise FileNotFoundError(f"Events file not found: {events_path}")

    # Treat paths without a file extension as output directories.
    if output_path.suffix == "" or output_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
        output_path = output_path / f"events.{fmt}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = _read_jsonl(events_path)

    if fmt.lower() == "csv":
        _write_csv(records, output_path)
    elif fmt.lower() == "jsonl":
        _write_jsonl(records, output_path)
    else:
        raise ValueError(f"Unsupported report format: {fmt}")

    logger.info("Exported report to %s", output_path)
    return output_path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_csv(records: list[dict[str, Any]], path: Path) -> None:
    if not records:
        path.write_text("")
        return
    fieldnames = list(records[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, default=str) + "\n")
