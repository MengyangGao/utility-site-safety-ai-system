"""Release-surface checks for committed demo assets and zone presets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from utility_safety_ai.zones.zone_loader import load_zones

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / "examples"


def test_all_committed_zone_presets_load_with_the_core_schema():
    zone_files = sorted(EXAMPLES.glob("zones*.yaml"))
    assert zone_files

    for path in zone_files:
        load_zones(path)


def test_asset_manifest_paths_and_hashes_match_committed_files():
    manifest = yaml.safe_load((EXAMPLES / "assets.yaml").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["assets"]

    for record in manifest["assets"]:
        path = EXAMPLES / record["path"]
        assert path.is_file(), f"Missing release asset: {record['path']}"
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == record["sha256"], f"Hash drift for {record['path']}"


def test_primary_documented_demo_inputs_exist():
    required = (
        "sample_images/construction_zone_01.jpg",
        "zones_construction_zone_01.yaml",
        "sample_images/construction_site_ppe_01.jpg",
        "zones_construction_site_ppe_01.yaml",
        "sample_videos/construction_rebar_pexels_10294768.mp4",
        "zones_construction_rebar_pexels_10294768.yaml",
    )

    for relative_path in required:
        assert (EXAMPLES / relative_path).is_file(), relative_path


def test_public_model_registry_is_machine_readable_and_honest():
    registry = json.loads((REPO_ROOT / "models" / "registry.json").read_text(encoding="utf-8"))

    assert registry["schema_version"] == "1.0"
    assert any(model["status"] == "clean-clone-default" for model in registry["models"])
    ppe = next(model for model in registry["models"] if "ppe" in model["capabilities"])
    assert len(ppe["sha256"]) == 64
    assert ppe["redistributable_with_repository"] is False
