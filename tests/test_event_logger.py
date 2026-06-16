"""Tests for event logger JSONL/CSV persistence."""

from __future__ import annotations

import csv
import json

from utility_safety_ai.events.event import SafetyEvent
from utility_safety_ai.events.event_logger import EventLogger


def _event(event_type: str = "missing_helmet", risk_level: str = "medium") -> SafetyEvent:
    return SafetyEvent(
        event_id="evt-1",
        timestamp="2026-01-01T00:00:00+00:00",
        source_type="image",
        source_path="test.jpg",
        frame_index=None,
        time_seconds=None,
        risk_level=risk_level,
        event_type=event_type,
        description="Test event",
        person_track_id=1,
        bbox=(10, 10, 30, 30),
        zone_id=None,
        zone_name=None,
        snapshot_path=None,
        metadata={"foo": "bar"},
    )


def test_logger_writes_jsonl_and_csv(tmp_path):
    logger = EventLogger(tmp_path)
    event = _event()
    logger.log(event)

    jsonl_file = tmp_path / "events.jsonl"
    csv_file = tmp_path / "events.csv"
    assert jsonl_file.exists()
    assert csv_file.exists()

    lines = jsonl_file.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["event_type"] == "missing_helmet"
    assert record["risk_level"] == "medium"

    rows = list(csv.DictReader(csv_file.read_text().splitlines()))
    assert len(rows) == 1
    assert rows[0]["event_type"] == "missing_helmet"


def test_logger_appends_multiple_events(tmp_path):
    logger = EventLogger(tmp_path)
    logger.log(_event(event_type="missing_helmet"))
    logger.log(_event(event_type="zone_intrusion", risk_level="high"))

    lines = (tmp_path / "events.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
