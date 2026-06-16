"""Tests for restricted-zone configuration loading."""

from __future__ import annotations

import json

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


def test_missing_file_returns_empty():
    zones = load_zones("/nonexistent/path/zones.yaml")
    assert zones == []


def test_none_path_returns_empty():
    zones = load_zones(None)
    assert zones == []
