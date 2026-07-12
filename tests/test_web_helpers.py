"""Tests for Streamlit helpers that do not require a real YOLO model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from utility_safety_ai.web_helpers import (
    ZoneValidationError,
    draw_zone_preview,
    inspect_detector,
    parse_zone_yaml,
    redact_uri_credentials,
    resolve_run_artifact,
    sanitize_run_artifacts,
    temporary_upload,
    zones_from_rows,
    zones_to_domain,
    zones_to_pixels,
    zones_to_rows,
    zones_to_yaml,
)
from utility_safety_ai.zones.zone_loader import load_zones

VALID_NORMALIZED = """
zones:
  - id: hv_1
    name: High Voltage
    risk_level: high
    coordinate_space: normalized
    required_ppe: [helmet, gloves]
    dwell_seconds: 2.5
    polygon:
      - [0.1, 0.2]
      - [0.8, 0.2]
      - [0.8, 0.9]
      - [0.1, 0.9]
"""


def test_normalized_zone_round_trip_and_core_conversion(tmp_path):
    zones = parse_zone_yaml(VALID_NORMALIZED)
    assert len(zones) == 1
    assert zones[0].polygon[0] == (0.1, 0.2)
    assert zones[0].required_ppe == ("helmet", "gloves")
    assert zones[0].dwell_seconds == 2.5

    serialized = zones_to_yaml(zones)
    schema = yaml.safe_load(serialized)
    assert "coordinate_space" not in schema
    assert schema["zones"][0]["coordinate_space"] == "normalized"
    assert schema["zones"][0]["required_ppe"] == ["helmet", "gloves"]
    assert schema["zones"][0]["dwell_seconds"] == 2.5
    reparsed = parse_zone_yaml(serialized)
    assert reparsed == zones

    core_zone = zones_to_domain(zones)[0]
    assert core_zone.coordinate_space == "normalized"
    assert core_zone.required_ppe == ("helmet", "gloves")
    assert core_zone.dwell_seconds == 2.5
    assert core_zone.resolved_polygon((1000, 500))[0] == (100.0, 100.0)

    pixel_zone = zones_to_pixels(zones, 1000, 500)[0]
    assert pixel_zone.coordinate_space == "pixels"
    assert pixel_zone.required_ppe == ("helmet", "gloves")
    assert pixel_zone.dwell_seconds == 2.5

    config_path = tmp_path / "zones.yaml"
    config_path.write_text(serialized, encoding="utf-8")
    loaded = load_zones(config_path)[0]
    assert loaded == core_zone


def test_legacy_pixel_zone_requires_dimensions_and_is_normalized():
    text = """
zones:
  - id: z1
    risk_level: medium
    polygon: [[100, 50], [900, 50], [900, 450], [100, 450]]
"""
    with pytest.raises(ZoneValidationError, match="source_width"):
        parse_zone_yaml(text)

    zone = parse_zone_yaml(text, source_width=1000, source_height=500)[0]
    assert zone.polygon == ((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9))


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (
            "zones:\n- id: z\n  coordinate_space: normalized\n  polygon: [[0,0],[2,0],[0,1]]",
            "between 0 and 1",
        ),
        (
            "zones:\n- id: z\n  coordinate_space: normalized\n  polygon: [[0,0],[0.5,0.5],[1,1]]",
            "area",
        ),
        (
            "zones:\n- id: z\n  coordinate_space: normalized\n  polygon: [[0,0],[1,0],[0,1]]\n- id: z\n  coordinate_space: normalized\n  polygon: [[0,0],[1,0],[0,1]]",
            "Duplicate",
        ),
    ],
)
def test_strict_zone_validation(text: str, message: str):
    with pytest.raises(ZoneValidationError, match=message):
        parse_zone_yaml(text)


def test_table_rows_round_trip():
    zones = parse_zone_yaml(VALID_NORMALIZED)
    rows = zones_to_rows(zones)
    assert zones_from_rows(rows) == zones


@pytest.mark.parametrize(
    ("text", "field"),
    [
        (
            "coordinate_space: normalized\nzones: []",
            "coordinate_space",
        ),
        (
            "zones:\n- id: typo\n  dwell_second: 2\n  polygon: [[0,0],[1,0],[0,1]]",
            "dwell_second",
        ),
        (
            "zones:\n- id: typo\n  required_PPE: [helmet]\n  polygon: [[0,0],[1,0],[0,1]]",
            "required_PPE",
        ),
    ],
)
def test_unknown_root_and_zone_fields_fail_fast(text: str, field: str):
    with pytest.raises(ZoneValidationError, match=field):
        parse_zone_yaml(text)


@pytest.mark.parametrize(
    ("policy", "message"),
    [
        ("required_ppe: helmet", "required_ppe must be a list"),
        ("required_ppe: [helmet, respirator]", "unsupported required PPE"),
        ("required_ppe: [helmet, helmet]", "must not contain duplicates"),
        ("dwell_seconds: '2.5'", "dwell_seconds must be numeric"),
        ("dwell_seconds: true", "dwell_seconds must be numeric"),
        ("dwell_seconds: -0.1", "finite non-negative"),
        ("dwell_seconds: .inf", "finite non-negative"),
    ],
)
def test_zone_policy_fields_are_strictly_validated(policy: str, message: str):
    text = f"""
zones:
  - id: strict
    coordinate_space: normalized
    {policy}
    polygon: [[0, 0], [1, 0], [0, 1]]
"""
    with pytest.raises(ZoneValidationError, match=message):
        parse_zone_yaml(text)


def test_zone_preview_draws_overlay_without_mutating_source():
    source = np.zeros((100, 200, 3), dtype=np.uint8)
    original = source.copy()
    rendered = draw_zone_preview(parse_zone_yaml(VALID_NORMALIZED), source)
    assert rendered.shape == source.shape
    assert np.array_equal(source, original)
    assert not np.array_equal(rendered, source)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("rtsp://worker:secret@camera.local/live", "rtsp://camera.local/live"),
        ("rtsp://worker:p@ss@camera.local/live", "rtsp://camera.local/live"),
        ("https://token@example.test/stream", "https://example.test/stream"),
        (
            "rtsp://user:secret@camera.local/live?token=abc&camera=west#private",
            "rtsp://camera.local/live?token=REDACTED&camera=west",
        ),
        (
            "https://example.test/live?api_key=a&auth=b&password=c&quality=hd#secret",
            "https://example.test/live?api_key=REDACTED&auth=REDACTED&password=REDACTED&quality=hd",
        ),
        ("0", "0"),
        ("/dev/video0", "/dev/video0"),
    ],
)
def test_redact_uri_credentials(source: str, expected: str):
    assert redact_uri_credentials(source) == expected


def test_sanitize_run_artifacts_replaces_secrets_in_errors_and_manifests(tmp_path):
    raw = (
        "rtsp://worker:hunter2@camera.local/live?api_key=abc123&camera=west"
        "#private-fragment"
    )
    redacted = redact_uri_credentials(raw)
    events = tmp_path / "events"
    events.mkdir()
    jsonl = events / "errors.jsonl"
    manifest = tmp_path / "web_run_manifest.json"
    binary = tmp_path / "snapshot.jpg"
    jsonl.write_text(f'{{"error": "Could not open {raw}"}}\n', encoding="utf-8")
    manifest.write_text(f'{{"source": "{raw}"}}\n', encoding="utf-8")
    binary.write_bytes(b"not an image, but no text secret")

    assert sanitize_run_artifacts(tmp_path, raw) == 2
    sanitized_text = jsonl.read_text(encoding="utf-8") + manifest.read_text(
        encoding="utf-8"
    )
    assert raw not in sanitized_text
    assert sanitized_text.count(redacted) == 2
    for secret in ("worker", "hunter2", "abc123", "private-fragment"):
        assert secret not in sanitized_text
    assert binary.read_bytes() == b"not an image, but no text secret"


class _FakeUpload:
    name = "camera.jpg"

    @staticmethod
    def getvalue() -> bytes:
        return b"temporary-image-bytes"


def test_temporary_upload_is_removed_even_when_consumer_fails():
    captured: Path | None = None
    with pytest.raises(RuntimeError, match="boom"), temporary_upload(_FakeUpload()) as path:
        captured = path
        assert path.exists()
        raise RuntimeError("boom")
    assert captured is not None
    assert not captured.exists()


class _FakeModel:
    names = {0: "person", 1: "helmet", 2: "no_helmet"}


class _FakeDetector:
    model = _FakeModel()
    device = "cpu"


def test_inspect_detector_reports_hash_classes_and_capabilities(tmp_path):
    model_path = tmp_path / "ppe.pt"
    model_path.write_bytes(b"fake-model")
    info = inspect_detector(_FakeDetector(), str(model_path))

    assert len(info["sha256"]) == 64
    assert info["requested_model"] == "ppe.pt"
    assert info["classes"] == ["person", "helmet", "no_helmet"]
    assert info["ppe_classes"] == ["helmet", "no_helmet"]
    assert set(info["capabilities"]) == {
        "person_detection",
        "zone_intrusion",
        "ppe_detection",
        "ppe_rule_events",
    }


def test_run_relative_artifact_is_resolved_without_persisting_host_path(tmp_path):
    expected = (tmp_path / "snapshots" / "event.jpg").resolve()
    assert resolve_run_artifact(tmp_path, "snapshots/event.jpg") == expected
    with pytest.raises(ValueError, match="escapes"):
        resolve_run_artifact(tmp_path, "../outside.jpg")
