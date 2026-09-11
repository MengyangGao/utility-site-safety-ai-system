"""Keep a reviewed demo exemption separate from ordinary input privacy."""

from dataclasses import replace
from pathlib import Path

from utility_safety_ai.monitoring.profiles import get_monitoring_profile
from utility_safety_ai.web.samples import load_sample_scenes, sample_settings, sample_zones
from utility_safety_ai.web.services import AnalysisSettings


def settings():
    return AnalysisSettings(
        "model.pt", "cpu", 0.25, 0.45, 10, True, "gaussian", get_monitoring_profile("balanced")
    )


def test_reviewed_scene_skips_only_its_own_redaction_without_changing_upload_settings():
    original = settings()
    for scene in load_sample_scenes():
        assert scene.matches_reviewed_image()
        chosen = sample_settings(original, scene, keep_clear=True)
        assert chosen.blur_faces is False
        assert chosen.privacy_reason.startswith("reviewed_rear_view_sample:")
        assert original.blur_faces is True
        assert sample_settings(original, scene, keep_clear=False) is original
        assert (
            sample_settings(original, replace(scene, no_visible_faces=False), keep_clear=True)
            is original
        )
    assert sample_zones(load_sample_scenes()[-1])


def test_changed_sample_does_not_inherit_the_reviewed_privacy_exemption(tmp_path: Path):
    original = settings()
    scene = load_sample_scenes()[0]
    changed = tmp_path / "changed.jpg"
    changed.write_bytes(b"different image content")
    assert sample_settings(original, replace(scene, image=changed), keep_clear=True) is original
