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
    confidence: float = 0.8,
) -> Detection:
    return Detection(
        class_id=0,
        class_name=class_name,
        confidence=confidence,
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
    assert records[0].negative_detections == []
    assert records[0].conflicting_detections == [no_helmet]


def test_positive_negative_resolution_is_independent_of_detection_order():
    person = _det("person", (0, 0, 100, 200), track_id=1)
    helmet = _det("helmet", (30, 10, 70, 50))
    no_helmet = _det("no_helmet", (35, 15, 65, 45))

    first, _ = associate_ppe_to_persons([person, helmet, no_helmet])
    second, _ = associate_ppe_to_persons([person, no_helmet, helmet])

    assert first[0].summary() == second[0].summary()
    assert first[0].violations() == second[0].violations() == []


def test_repeated_negative_boxes_are_one_resolved_violation():
    person = _det("person", (0, 0, 100, 200), track_id=1)
    lower_confidence = _det("no_gloves", (5, 80, 35, 130), confidence=0.6)
    higher_confidence = _det("no_gloves", (65, 80, 95, 130), confidence=0.9)

    records, _ = associate_ppe_to_persons(
        [person, lower_confidence, higher_confidence]
    )

    assert records[0].violations() == ["gloves"]
    assert records[0].negative_detections == [higher_confidence]


def test_multiple_untracked_people_do_not_overwrite_each_other():
    records, _ = associate_ppe_to_persons(
        [
            _det("person", (0, 0, 50, 100)),
            _det("person", (100, 0, 150, 100)),
        ]
    )
    assert len(records) == 2
    assert [record.bbox for record in records] == [
        (0, 0, 50, 100),
        (100, 0, 150, 100),
    ]


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
    assert "conflicting_classes" in text
