"""Release-surface checks for committed demo assets and zone presets."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from utility_safety_ai.governance import audit_provenance
from utility_safety_ai.zones.zone_loader import load_zones

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPO_ROOT / "examples"


def test_all_committed_zone_presets_load_with_the_core_schema():
    zone_files = sorted(EXAMPLES.glob("zones*.yaml"))
    assert zone_files

    for path in zone_files:
        load_zones(path)


def test_asset_manifest_paths_and_hashes_match_committed_files():
    manifest = REPO_ROOT / "docs" / "legal" / "provenance.yaml"
    report = audit_provenance(manifest, REPO_ROOT)

    assert report["passed"], report["errors"]
    assert report["artifact_count"] == 25
    assert report["approved_licenses"] == ["AGPL-3.0-only", "CC0-1.0"]


def test_primary_documented_demo_inputs_exist():
    required = (
        "sample_images/construction_site_ppe_01.jpg",
        "zones_construction_site_ppe_01.yaml",
        "sample_videos/construction_ppe_pan.mp4",
    )

    for relative_path in required:
        assert (EXAMPLES / relative_path).is_file(), relative_path


def test_public_model_registry_is_machine_readable_and_honest():
    registry = json.loads((REPO_ROOT / "models" / "registry.json").read_text(encoding="utf-8"))

    assert registry["schema_version"] == "1.0"
    assert any(model["status"] == "bundled-default" for model in registry["models"])
    ppe = next(model for model in registry["models"] if "ppe" in model["capabilities"])
    assert len(ppe["sha256"]) == 64
    assert ppe["redistributable_with_repository"] is True
    assert ppe["license"] == "AGPL-3.0-only"
    assert (REPO_ROOT / "models" / ppe["filename"]).stat().st_size < 10_000_000


def test_tracked_root_is_minimal():
    tracked = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    root_files = {
        path for path in tracked if "/" not in path and (REPO_ROOT / path).is_file()
    }

    assert root_files == {".gitignore", "LICENSE", "README.md", "pyproject.toml", "requirements.txt"}
