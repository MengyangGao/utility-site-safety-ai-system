"""Tests for person-PPE association and compliance reporting."""

from __future__ import annotations

from utility_safety_ai.compliance.compliance_reporter import ComplianceReporter
from utility_safety_ai.compliance.person_ppe_association import (
    associate_ppe_to_persons,
)
from utility_safety_ai.events.event import Detection


def _det(
    class_name: str,
    bbox: tuple[float, float, float, float],
    track_id: int | None = None,
) -> Detection:
    return Detection(
        class_id=0,
        class_name=class_name,
        confidence=0.8,
        bbox=bbox,
        track_id=track_id,
    )


def test_positive_ppe_inside_person_is_associated():
    person = _det("person", (0, 0, 100, 200), track_id=1)
    helmet = _det("helmet", (30, 10, 70, 50), track_id=2)
    records, unassociated = associate_ppe_to_persons([person, helmet])
    assert len(records) == 1
    assert records[0].items["helmet"] == "yes"
    assert len(unassociated) == 0


def test_negative_ppe_inside_person_is_associated():
    person = _det("person", (0, 0, 100, 200), track_id=1)
    no_vest = _det("no_vest", (20, 80, 80, 180), track_id=2)
    records, unassociated = associate_ppe_to_persons([person, no_vest])
    assert len(records) == 1
    assert records[0].items["vest"] == "no"
    assert records[0].violations() == ["vest"]
    assert len(unassociated) == 0


def test_positive_ppe_takes_precedence_over_negative():
    person = _det("person", (0, 0, 100, 200), track_id=1)
    helmet = _det("helmet", (30, 10, 70, 50), track_id=2)
    no_helmet = _det("no_helmet", (35, 15, 65, 45), track_id=3)
    records, _ = associate_ppe_to_persons([person, helmet, no_helmet])
    assert records[0].items["helmet"] == "yes"


def test_unassociated_ppe_returned_separately():
    person = _det("person", (0, 0, 100, 100), track_id=1)
    helmet = _det("helmet", (200, 200, 250, 250), track_id=2)
    records, unassociated = associate_ppe_to_persons([person, helmet])
    assert len(records) == 1
    assert records[0].items["helmet"] == "unknown"
    assert len(unassociated) == 1


def test_compliance_reporter_writes_csv_and_jsonl(tmp_path):
    records, _ = associate_ppe_to_persons(
        [
            _det("person", (0, 0, 100, 200), track_id=1),
            _det("helmet", (30, 10, 70, 50), track_id=2),
            _det("no_vest", (20, 80, 80, 180), track_id=3),
        ]
    )
    reporter = ComplianceReporter(tmp_path)
    jsonl_path, csv_path = reporter.write(records, frame_index=0, time_seconds=0.0)
    assert jsonl_path.exists()
    assert csv_path.exists()
    text = csv_path.read_text()
    assert "person_track_id" in text
    assert "helmet" in text
    assert "vest" in text
