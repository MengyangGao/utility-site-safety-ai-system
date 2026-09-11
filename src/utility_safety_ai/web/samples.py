"""Reviewed sample scenes and their explicit privacy context."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path

import yaml
from PIL import Image

from ..resources import resource_root
from ..web_helpers import NormalizedZone, parse_zone_yaml
from .services import AnalysisSettings


@dataclass(frozen=True)
class SampleScene:
    id: str
    title: str
    image: Path
    zones: Path
    description: str
    attribution: str
    source_url: str
    license: str
    sha256: str
    no_visible_faces: bool = False

    def matches_reviewed_image(self) -> bool:
        return (
            self.image.is_file()
            and hashlib.sha256(self.image.read_bytes()).hexdigest() == self.sha256
        )


def load_sample_scenes() -> list[SampleScene]:
    root = resource_root() / "examples"
    manifest = root / "demo_scenes.yaml"
    if not manifest.is_file():
        return []
    records = yaml.safe_load(manifest.read_text(encoding="utf-8"))["scenes"]
    scenes = []
    for record in records:
        payload = dict(record)
        for field in ("image", "zones"):
            path = (root / payload[field]).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError("Sample assets must stay inside the examples directory")
            payload[field] = path
        scenes.append(SampleScene(**payload))
    return scenes


def sample_settings(
    settings: AnalysisSettings, scene: SampleScene, *, keep_clear: bool
) -> AnalysisSettings:
    """Only the exact reviewed rear-view asset can skip the configured redaction."""
    if keep_clear and scene.no_visible_faces and scene.matches_reviewed_image():
        return replace(
            settings,
            blur_faces=False,
            privacy_reason=f"reviewed_rear_view_sample:{scene.id}:no_visible_faces",
        )
    return settings


def sample_zones(scene: SampleScene) -> list[NormalizedZone]:
    with Image.open(scene.image) as image:
        width, height = image.size
    return parse_zone_yaml(
        scene.zones.read_text(encoding="utf-8"), source_width=width, source_height=height
    )
