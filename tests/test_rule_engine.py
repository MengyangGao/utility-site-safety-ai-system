"""Tests for the safety rule engine."""

from __future__ import annotations

from utility_safety_ai.events.event import Detection
from utility_safety_ai.rules import risk
from utility_safety_ai.rules.rule_engine import RuleEngine
from utility_safety_ai.zones.zone import Zone

ZONES = [
    Zone(id="zone1", name="High Voltage", risk_level=risk.HIGH, polygon=[(0, 0), (100, 0), (100, 100), (0, 100)]),
]


def _det(class_name: str, bbox: tuple[float, float, float, float], track_id: int | None = None) -> Detection:
    return Detection(class_id=0, class_name=class_name, confidence=0.8, bbox=bbox, track_id=track_id)


def test_missing_helmet_is_medium_risk():
    engine = RuleEngine(zones=[])
    events = engine.evaluate(
        [_det("no_helmet", (10, 10, 30, 30), track_id=1)],
        source_type="image",
        source_path="test.jpg",
    )
    assert len(events) == 1
    assert events[0].event_type == "missing_helmet"
    assert events[0].risk_level == risk.MEDIUM


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


def test_cooldown_suppresses_duplicate_events():
    engine = RuleEngine(zones=ZONES, cooldown_seconds=1.0)
    detections = [_det("person", (10, 10, 30, 30), track_id=1)]
    first = engine.evaluate(detections, source_type="video", source_path="test.mp4", time_seconds=0.0)
    second = engine.evaluate(detections, source_type="video", source_path="test.mp4", time_seconds=0.5)
    assert len(first) == 1
    assert len(second) == 0


def test_cooldown_allows_event_after_window():
    engine = RuleEngine(zones=ZONES, cooldown_seconds=1.0)
    detections = [_det("person", (10, 10, 30, 30), track_id=1)]
    first = engine.evaluate(detections, source_type="video", source_path="test.mp4", time_seconds=0.0)
    second = engine.evaluate(detections, source_type="video", source_path="test.mp4", time_seconds=1.5)
    assert len(first) == 1
    assert len(second) == 1
