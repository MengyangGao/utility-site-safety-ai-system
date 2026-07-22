"""Tests for monitoring profiles and transparent run diagnostics."""

from __future__ import annotations

import json

import pytest

from utility_safety_ai.events.event import Detection
from utility_safety_ai.monitoring import MonitoringQuality, get_monitoring_profile
from utility_safety_ai.rules.rule_engine import RuleEngine


def _det(name: str, bbox, track_id=None):
    return Detection(0, name, 0.9, bbox, track_id)


def test_profiles_encode_distinct_quality_tradeoffs():
    precise = get_monitoring_profile("high_precision")
    sensitive = get_monitoring_profile("high_sensitivity")
    assert precise.confidence > sensitive.confidence
    assert precise.ppe_confirmation_frames > sensitive.ppe_confirmation_frames
    with pytest.raises(ValueError, match="Unknown monitoring profile"):
        get_monitoring_profile("magic")


def test_quality_summary_and_artifacts_are_explicit(tmp_path):
    quality = MonitoringQuality()
    engine = RuleEngine(
        zones=[],
        rules_config={"confirmation_frames": {"default": 2}},
    )
    detections = [
        _det("person", (0, 0, 100, 200), 7),
        _det("helmet", (30, 5, 70, 45)),
        _det("no_vest", (200, 200, 250, 260)),
    ]
    evaluation = engine.evaluate_frame(
        detections, "video", "clip.mp4", time_seconds=0.0
    )
    quality.observe(detections, evaluation, inference_seconds=0.02, pipeline_seconds=0.04)

    summary = quality.summary()
    assert summary["tracked_person_rate"] == 1.0
    assert summary["ppe_assignment_rate"] == 0.5
    assert summary["unique_track_ids"] == 1
    assert "not identity accuracy" in summary["interpretation"]["tracked_person_rate"]

    json_path, csv_path = quality.write(tmp_path)
    assert json.loads(json_path.read_text())["frames_processed"] == 1
    assert "ppe_assignment_rate" in csv_path.read_text()
