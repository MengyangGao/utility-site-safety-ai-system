"""Aggregate safety events into a compact summary report."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .event import SafetyEvent


def aggregate_events(events: list[SafetyEvent]) -> dict[str, Any]:
    """Return a summary dict with counts by risk level and event type.

    Args:
        events: List of emitted safety events.

    Returns:
        Dictionary suitable for JSON/CSV serialization.
    """
    risk_counts = Counter(e.risk_level for e in events)
    type_counts = Counter(e.event_type for e in events)
    zone_intrusions = sum(1 for e in events if e.event_type == "zone_intrusion")
    ppe_violations = sum(
        1 for e in events if e.event_type.startswith("missing_")
    )
    unique_persons = {e.person_track_id for e in events if e.person_track_id is not None}

    timestamps = [e.timestamp for e in events if e.timestamp]
    return {
        "total_events": len(events),
        "unique_persons": len(unique_persons),
        "zone_intrusions": zone_intrusions,
        "ppe_violations": ppe_violations,
        "by_risk_level": dict(risk_counts),
        "by_event_type": dict(type_counts),
        "first_timestamp": min(timestamps) if timestamps else None,
        "last_timestamp": max(timestamps) if timestamps else None,
    }


def write_summary(events: list[SafetyEvent], output_dir: str | Path) -> Path:
    """Write ``summary.json`` and ``summary.csv`` into the events directory.

    Args:
        events: List of emitted safety events.
        output_dir: Directory where the events logs are stored.

    Returns:
        Path to the written JSON summary.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = aggregate_events(events)

    json_path = output_dir / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["total_events", summary["total_events"]])
        writer.writerow(["unique_persons", summary["unique_persons"]])
        writer.writerow(["zone_intrusions", summary["zone_intrusions"]])
        writer.writerow(["ppe_violations", summary["ppe_violations"]])
        for level, count in summary["by_risk_level"].items():
            writer.writerow([f"risk_{level}", count])
        for event_type, count in summary["by_event_type"].items():
            writer.writerow([f"type_{event_type}", count])

    return json_path
