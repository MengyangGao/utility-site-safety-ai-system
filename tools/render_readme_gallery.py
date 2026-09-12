"""Reproduce the README's real construction-site detection outputs, locally."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import yaml

from utility_safety_ai.detection.yolo_detector import YoloDetector
from utility_safety_ai.pipelines.image_pipeline import run_image_pipeline
from utility_safety_ai.zones.zone_loader import load_zones

ROOT = Path(__file__).resolve().parents[1]
SCENES = [
    ("hk_site_access_01", "Hong Kong building site", None, "LicenseRef-Unsplash"),
    ("hk_tsuen_wan_scaffolding_01", "Tsuen Wan, Hong Kong", None, "LicenseRef-Unsplash"),
    (
        "piling_rear_view_01",
        "Piling works (city not verified)",
        "zones_piling_rear_view.yaml",
        "LicenseRef-Pexels",
    ),
]


def main() -> None:
    detector = YoloDetector(
        model_path=str(ROOT / "models/ppe_yolo11n.pt"), device="cpu", conf=0.25, iou=0.45
    )
    registry_path = ROOT / "docs/legal/provenance.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    registered = {record["path"]: record for record in registry["artifacts"]}
    output = ROOT / "outputs/readme-gallery"
    records = []
    for name, location, zone_file, license_id in SCENES:
        source = ROOT / "examples/sample_images" / f"{name}.jpg"
        source_name = source.relative_to(ROOT).as_posix()
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        if registered[source_name]["sha256"] != source_hash:
            raise ValueError(f"Source image changed: {source_name}")
        zones = load_zones(ROOT / "examples" / zone_file) if zone_file else []
        run_image_pipeline(
            source,
            output,
            detector,
            zones,
            blur_faces_enabled=False,
            privacy_reason="publicly_licensed_readme_demonstration_unredacted",
        )
        latest = json.loads((output / "latest.json").read_text(encoding="utf-8"))
        run_dir = output / latest["run_dir"]
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        relative = f"docs/assets/{name}_detections.jpg"
        shutil.copyfile(next((run_dir / "images").glob("*")), ROOT / relative)
        image_hash = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        record = registered.get(relative)
        if record is None:
            record = {"path": relative}
            registry["artifacts"].append(record)
        record.update(
            sha256=image_hash,
            license=license_id,
            derived_from=source_name,
            generated_by="tools/render_readme_gallery.py; real bundled PPE model",
            privacy_redaction="disabled_for_publicly_licensed_readme_example",
        )
        records.append(
            {
                "image": relative,
                "location": location,
                "source_sha256": source_hash,
                "image_sha256": image_hash,
                "model_sha256": manifest["model"]["model_sha256"],
                "metrics": manifest["metrics"],
                "privacy_blur_enabled": False,
                "run": run_dir.relative_to(ROOT).as_posix(),
            }
        )
    registry_path.write_text(
        yaml.safe_dump(registry, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    report = {
        "scope": "Real photo/model outputs, not ground-truth accuracy or verified violations.",
        "confidence": 0.25,
        "device": "cpu",
        "scenes": records,
    }
    report_path = ROOT / "docs/validation/readme-gallery.json"
    if report_path.exists():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        report["review_notes"] = previous.get("review_notes", {})
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
