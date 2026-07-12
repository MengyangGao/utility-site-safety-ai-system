"""Tests for restricted-zone configuration loading."""

from __future__ import annotations

import json

import pytest
import yaml

from utility_safety_ai.zones.zone_loader import load_zones


def test_load_yaml_zones(tmp_path):
    path = tmp_path / "zones.yaml"
    data = {
        "zones": [
            {
                "id": "z1",
                "name": "Zone One",
                "risk_level": "high",
                "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
            }
        ]
    }
    path.write_text(yaml.safe_dump(data))

    zones = load_zones(path)
    assert len(zones) == 1
    assert zones[0].id == "z1"
    assert zones[0].name == "Zone One"
    assert zones[0].risk_level == "high"
    assert zones[0].polygon == [(0, 0), (10, 0), (10, 10), (0, 10)]


def test_load_json_zones(tmp_path):
    path = tmp_path / "zones.json"
    data = {
        "zones": [
            {
                "id": "z2",
                "name": "Zone Two",
                "risk_level": "medium",
                "polygon": [[0, 0], [5, 0], [5, 5], [0, 5]],
            }
        ]
    }
    path.write_text(json.dumps(data))

    zones = load_zones(path)
    assert len(zones) == 1
    assert zones[0].id == "z2"
    assert zones[0].risk_level == "medium"


def test_explicit_missing_file_fails_fast():
    with pytest.raises(FileNotFoundError, match="not found"):
        load_zones("/nonexistent/path/zones.yaml")


def test_none_path_returns_empty():
    zones = load_zones(None)
    assert zones == []


def test_load_normalized_zone_policy(tmp_path):
    path = tmp_path / "zones.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "zones": [
                    {
                        "id": "z1",
                        "name": "Policy Zone",
                        "risk_level": "critical",
                        "coordinate_space": "normalized",
                        "polygon": [[0.1, 0.2], [0.9, 0.2], [0.9, 0.8]],
                        "required_ppe": ["helmet", "gloves"],
                        "dwell_seconds": 1.5,
                    }
                ]
            }
        )
    )

    zone = load_zones(path)[0]
    assert zone.coordinate_space == "normalized"
    assert zone.required_ppe == ("helmet", "gloves")
    assert zone.dwell_seconds == 1.5
    assert zone.resolved_polygon((100, 200))[0] == (10, 40)


@pytest.mark.parametrize(
    "data, message",
    [
        ({"zones": "not-a-list"}, "must be a list"),
        ({"zones": [{"id": "z1", "polygon": [[0, 0], [1, 1]]}]}, "at least 3"),
        (
            {
                "zones": [
                    {
                        "id": "z1",
                        "polygon": [[0, 0], [1, 0], [1, 1]],
                        "unexpected": True,
                    }
                ]
            },
            "unknown fields",
        ),
        (
            {
                "zones": [
                    {"id": "z1", "polygon": [[0, 0], [1, 0], [1, 1]]},
                    {"id": "z1", "polygon": [[0, 0], [1, 0], [1, 1]]},
                ]
            },
            "Duplicate zone id",
        ),
    ],
)
def test_invalid_zone_schema_fails_fast(tmp_path, data, message):
    path = tmp_path / "zones.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match=message):
        load_zones(path)
