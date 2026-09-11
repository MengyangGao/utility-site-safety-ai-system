"""Tests for the safety rule engine."""

from __future__ import annotations

import pytest

from utility_safety_ai.events.event import Detection
from utility_safety_ai.rules import risk
from utility_safety_ai.rules.rule_engine import RuleEngine
from utility_safety_ai.zones.zone import Zone

ZONES = [
    Zone(
        id="zone1",
        name="High Voltage",
        risk_level=risk.HIGH,
        polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
    ),
]


def _det(
    class_name: str, bbox: tuple[float, float, float, float], track_id: int | None = None
) -> Detection:
    return Detection(
        class_id=0, class_name=class_name, confidence=0.8, bbox=bbox, track_id=track_id
    )


def test_missing_helmet_is_medium_risk():
    engine = RuleEngine(zones=[])
    events = engine.evaluate(
        [
            _det("person", (0, 0, 40, 80), track_id=1),
            _det("no_helmet", (10, 10, 30, 30)),
        ],
        source_type="image",
        source_path="test.jpg",
    )
    assert len(events) == 1
    assert events[0].event_type == "missing_helmet"
    assert events[0].risk_level == risk.MEDIUM


def test_unassociated_negative_ppe_is_audited_but_not_an_event():
    engine = RuleEngine(zones=[])

    evaluation = engine.evaluate_frame(
        [_det("no_helmet", (10, 10, 30, 30))],
        source_type="image",
        source_path="test.jpg",
    )

    assert evaluation.active_findings == []
    assert evaluation.new_events == []


def test_zone_intrusion_is_high_risk():
    engine = RuleEngine(zones=ZONES)
    events = engine.evaluate(
        [_det("person", (10, 10, 30, 30), track_id=1)],
        source_type="image",
        source_path="test.jpg",
    )
    assert len(events) == 1
    assert events[0].event_type == "zone_intrusion"
    assert events[0].risk_level == risk.HIGH
    assert events[0].zone_id == "zone1"


def test_missing_helmet_in_zone_is_critical():
    engine = RuleEngine(zones=ZONES)
    events = engine.evaluate(
        [
            _det("person", (10, 10, 30, 30), track_id=1),
            _det("no_helmet", (12, 12, 28, 28), track_id=1),
        ],
        source_type="image",
        source_path="test.jpg",
    )
    zone_events = [e for e in events if e.event_type == "zone_intrusion"]
    assert len(zone_events) == 1
    assert zone_events[0].risk_level == risk.CRITICAL


def test_multiple_ppe_violations_escalate_to_high():
    engine = RuleEngine(zones=[])
    events = engine.evaluate(
        [
            _det("person", (0, 0, 50, 50), track_id=1),
            _det("no_helmet", (15, 5, 35, 25), track_id=2),
            _det("no_vest", (10, 20, 40, 45), track_id=3),
        ],
        source_type="image",
        source_path="test.jpg",
    )
    assert len(events) == 2
    assert all(e.risk_level == risk.HIGH for e in events)


def test_positive_negative_conflict_does_not_emit_missing_event():
    engine = RuleEngine(zones=ZONES)
    events = engine.evaluate(
        [
            _det("person", (10, 10, 30, 30), track_id=1),
            _det("no_helmet", (12, 12, 28, 28)),
            _det("helmet", (12, 12, 28, 28)),
        ],
        source_type="image",
        source_path="test.jpg",
    )

    assert [event.event_type for event in events] == ["zone_intrusion"]
    assert events[0].risk_level == risk.HIGH


def test_duplicate_negative_boxes_count_as_one_ppe_type():
    engine = RuleEngine(zones=[])
    events = engine.evaluate(
        [
            _det("person", (0, 0, 100, 200), track_id=1),
            _det("no_gloves", (5, 80, 35, 130)),
            _det("no_gloves", (65, 80, 95, 130)),
        ],
        source_type="image",
        source_path="test.jpg",
    )

    assert len(events) == 1
    assert events[0].event_type == "missing_gloves"
    assert events[0].risk_level == risk.LOW


def test_cooldown_suppresses_duplicate_events():
    engine = RuleEngine(zones=ZONES, cooldown_seconds=1.0)
    detections = [_det("person", (10, 10, 30, 30), track_id=1)]
    first = engine.evaluate(
        detections, source_type="video", source_path="test.mp4", time_seconds=0.0
    )
    second = engine.evaluate(
        detections, source_type="video", source_path="test.mp4", time_seconds=0.5
    )
    assert len(first) == 1
    assert len(second) == 0


def test_cooldown_allows_event_after_window():
    engine = RuleEngine(zones=ZONES, cooldown_seconds=1.0)
    detections = [_det("person", (10, 10, 30, 30), track_id=1)]
    first = engine.evaluate(
        detections, source_type="video", source_path="test.mp4", time_seconds=0.0
    )
    second = engine.evaluate(
        detections, source_type="video", source_path="test.mp4", time_seconds=1.5
    )
    assert len(first) == 1
    assert len(second) == 1


def test_cooldown_survives_tracker_id_switch_for_overlapping_person():
    engine = RuleEngine(zones=ZONES, cooldown_seconds=10.0)
    first = engine.evaluate(
        [_det("person", (10, 10, 60, 90), track_id=1)],
        source_type="video",
        source_path="test.mp4",
        time_seconds=0.0,
    )
    switched = engine.evaluate(
        [_det("person", (12, 12, 58, 88), track_id=99)],
        source_type="video",
        source_path="test.mp4",
        time_seconds=1.0,
    )

    assert len(first) == 1
    assert switched == []


def test_spatial_cooldown_does_not_merge_separate_people():
    wide_zone = Zone(
        id="wide",
        name="Wide",
        risk_level="high",
        polygon=[(0, 0), (300, 0), (300, 100), (0, 100)],
    )
    engine = RuleEngine(zones=[wide_zone], cooldown_seconds=10.0)
    first = engine.evaluate(
        [_det("person", (10, 10, 40, 80), track_id=1)],
        "video",
        "test.mp4",
        time_seconds=0.0,
    )
    second = engine.evaluate(
        [_det("person", (200, 10, 240, 80), track_id=2)],
        "video",
        "test.mp4",
        time_seconds=1.0,
    )

    assert len(first) == len(second) == 1


def test_frame_evaluation_separates_active_findings_from_new_events():
    engine = RuleEngine(zones=ZONES, cooldown_seconds=10.0)
    detections = [_det("person", (10, 10, 30, 30), track_id=1)]

    first = engine.evaluate_frame(
        detections,
        source_type="video",
        source_path="test.mp4",
        time_seconds=0.0,
    )
    repeated = engine.evaluate_frame(
        detections,
        source_type="video",
        source_path="test.mp4",
        time_seconds=1.0,
    )

    assert len(first.active_findings) == len(first.new_events) == 1
    assert first.active_findings[0].metadata["lifecycle_state"] == "opened"
    assert len(repeated.active_findings) == 1
    assert repeated.active_findings[0].metadata["lifecycle_state"] == "ongoing"
    assert repeated.new_events == []

    cleared = engine.evaluate_frame(
        [],
        source_type="video",
        source_path="test.mp4",
        time_seconds=2.0,
    )
    assert cleared.active_findings == []
    assert cleared.resolved_findings[0].metadata["lifecycle_state"] == "resolved"


def test_zone_dwell_delays_intrusion_until_threshold():
    dwell_zone = Zone(
        id="dwell",
        name="Dwell Zone",
        risk_level="high",
        polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
        dwell_seconds=2.0,
    )
    engine = RuleEngine(zones=[dwell_zone])
    detections = [_det("person", (10, 10, 30, 30), track_id=1)]

    assert engine.evaluate(detections, "video", "test.mp4", time_seconds=0.0) == []
    assert engine.evaluate(detections, "video", "test.mp4", time_seconds=1.9) == []
    events = engine.evaluate(detections, "video", "test.mp4", time_seconds=2.0)

    assert len(events) == 1
    assert events[0].metadata["time_in_zone_seconds"] == 2.0


def test_video_ppe_event_requires_consecutive_confirmation():
    engine = RuleEngine(
        zones=[],
        rules_config={"confirmation_frames": {"default": 3}},
    )
    detections = [
        _det("person", (0, 0, 40, 80), track_id=1),
        _det("no_helmet", (10, 4, 30, 25)),
    ]

    first = engine.evaluate_frame(detections, "video", "test.mp4", time_seconds=0.0)
    second = engine.evaluate_frame(detections, "video", "test.mp4", time_seconds=0.1)
    third = engine.evaluate_frame(detections, "video", "test.mp4", time_seconds=0.2)

    assert first.new_events == second.new_events == []
    assert first.provisional_findings[0].metadata["confirmation_count"] == 1
    assert second.provisional_findings[0].metadata["confirmation_count"] == 2
    assert third.provisional_findings == []
    assert len(third.confirmed_findings) == len(third.new_events) == 1
    assert third.new_events[0].metadata["confirmation_count"] == 3


def test_provisional_false_positive_resets_before_confirmation():
    engine = RuleEngine(
        zones=[],
        rules_config={"confirmation_frames": {"default": 3}},
    )
    detections = [
        _det("person", (0, 0, 40, 80), track_id=1),
        _det("no_helmet", (10, 4, 30, 25)),
    ]

    assert engine.evaluate(detections, "video", "test.mp4", time_seconds=0.0) == []
    cleared = engine.evaluate_frame([], "video", "test.mp4", time_seconds=0.1)
    assert cleared.resolved_findings[0].metadata["confirmation_status"] == "observing"
    restarted = engine.evaluate_frame(detections, "video", "test.mp4", time_seconds=0.2)
    assert restarted.active_findings[0].metadata["confirmation_count"] == 1


def test_image_evidence_remains_immediate_with_temporal_profile():
    engine = RuleEngine(
        zones=[],
        rules_config={"confirmation_frames": {"default": 5}},
    )
    events = engine.evaluate(
        [
            _det("person", (0, 0, 40, 80), track_id=1),
            _det("no_helmet", (10, 4, 30, 25)),
        ],
        "image",
        "test.jpg",
    )

    assert len(events) == 1
    assert events[0].metadata["confirmation_required"] == 1


def test_zone_dwell_resets_after_person_leaves():
    dwell_zone = Zone(
        id="dwell",
        name="Dwell Zone",
        risk_level="high",
        polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
        dwell_seconds=1.0,
    )
    engine = RuleEngine(zones=[dwell_zone])
    inside = [_det("person", (10, 10, 30, 30), track_id=1)]

    engine.evaluate(inside, "video", "test.mp4", time_seconds=0.0)
    engine.evaluate([], "video", "test.mp4", time_seconds=0.5)
    assert engine.evaluate(inside, "video", "test.mp4", time_seconds=1.1) == []


def test_required_ppe_policy_can_disable_non_required_violation():
    engine = RuleEngine(zones=[], rules_config={"required_ppe": ["helmet"]})
    events = engine.evaluate(
        [
            _det("person", (0, 0, 100, 200), track_id=1),
            _det("no_gloves", (5, 80, 35, 130)),
            _det("no_helmet", (30, 5, 70, 50)),
        ],
        source_type="image",
        source_path="test.jpg",
    )
    assert [event.event_type for event in events] == ["missing_helmet"]


def test_normalized_zone_uses_frame_size_metadata():
    zone = Zone(
        id="normalized",
        name="Normalized",
        risk_level="high",
        coordinate_space="normalized",
        polygon=[(0.5, 0.5), (1, 0.5), (1, 1), (0.5, 1)],
    )
    engine = RuleEngine(zones=[zone])
    events = engine.evaluate(
        [_det("person", (60, 60, 80, 80), track_id=1)],
        source_type="image",
        source_path="test.jpg",
        metadata={"frame_size": (100, 100)},
    )
    assert [event.event_type for event in events] == ["zone_intrusion"]


def test_non_finite_event_time_is_rejected():
    engine = RuleEngine(zones=[])
    with pytest.raises(ValueError, match="time_seconds"):
        engine.evaluate([], "video", "test.mp4", time_seconds=float("nan"))


def test_incident_identity_is_stable_until_observation_ends():
    engine = RuleEngine(zones=ZONES, cooldown_seconds=0)
    detections = [_det("person", (10, 10, 30, 30), track_id=1)]
    first = engine.evaluate_frame(detections, "camera", "test", time_seconds=0)
    second = engine.evaluate_frame(detections, "camera", "test", time_seconds=1)
    assert (
        first.active_findings[0].metadata["incident_id"]
        == second.active_findings[0].metadata["incident_id"]
    )
    ended = engine.end_stream("connection_lost")
    assert ended[0].metadata["lifecycle_state"] == "interrupted"
    assert ended[0].metadata["interruption_reason"] == "connection_lost"
    third = engine.evaluate_frame(detections, "camera", "test", time_seconds=2)
    assert (
        third.active_findings[0].metadata["incident_id"]
        != first.active_findings[0].metadata["incident_id"]
    )


def test_cooldown_memory_expires_during_blank_frames():
    engine = RuleEngine(zones=ZONES, cooldown_seconds=10)
    engine.evaluate(
        [_det("person", (10, 10, 30, 30), track_id=1)], "camera", "test", time_seconds=0
    )
    engine.evaluate_frame([], "camera", "test", time_seconds=11)
    assert not engine._last_emitted
    assert not engine._recent_spatial_events


def test_distinct_untracked_people_are_not_the_same_incident():
    engine = RuleEngine(zones=ZONES)
    events = engine.evaluate(
        [_det("person", (1, 1, 20, 50)), _det("person", (60, 1, 90, 50))],
        "camera",
        "test",
        time_seconds=0,
    )
    assert len(events) == 2
