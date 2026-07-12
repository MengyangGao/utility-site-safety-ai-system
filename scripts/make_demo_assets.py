"""Generate derived demo videos without mutating committed release assets.

The script reads provenance-tracked source images from ``examples/`` and writes
derived files below ``outputs/generated_demo_assets`` by default. Committed
example hashes therefore remain stable across OpenCV/codec versions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_IMAGES = REPO_ROOT / "examples" / "sample_images"
DEMO_IMAGE = SAMPLE_IMAGES / "construction_zone_01.jpg"


def _make_construction_zones(width: int, height: int) -> dict:
    """Return localized hazard zones for the construction-zone demo image."""
    lift = {
        "id": "aerial_lift_platform",
        "name": "Aerial Lift / Live Work Zone",
        "risk_level": "high",
        "polygon": [
            [int(width * 0.30), int(height * 0.825)],
            [int(width * 0.72), int(height * 0.825)],
            [int(width * 0.75), int(height * 0.994)],
            [int(width * 0.27), int(height * 0.994)],
        ],
    }
    tape = {
        "id": "danger_tape_boundary",
        "name": "Danger Tape Boundary",
        "risk_level": "medium",
        "polygon": [
            [0, int(height * 0.95)],
            [width, int(height * 0.95)],
            [width, height],
            [0, height],
        ],
    }
    return {"zones": [lift, tape]}


def _create_ken_burns_video(
    image_path: Path,
    output_path: Path,
    duration: float = 5.0,
    fps: int = 15,
    zoom_start: float = 1.0,
    zoom_end: float = 1.12,
) -> None:
    """Create a slow pan/zoom video from a static image to simulate CCTV footage."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    height, width = image.shape[:2]
    frame_count = int(duration * fps)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video writer: {output_path}")

    for i in range(frame_count):
        t = i / max(1, frame_count - 1)
        zoom = zoom_start + (zoom_end - zoom_start) * t
        # Pan slightly from left to right as we zoom in.
        pan_x = (zoom - 1.0) * width * 0.15 * t
        crop_w = int(width / zoom)
        crop_h = int(height / zoom)
        x1 = int(min(pan_x, width - crop_w))
        y1 = int((height - crop_h) / 2)
        x2 = x1 + crop_w
        y2 = y1 + crop_h
        cropped = image[y1:y2, x1:x2]
        frame = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
        writer.write(frame)

    writer.release()
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Generated video is empty: {output_path}")


def _create_vertical_pan_video(
    image_path: Path,
    output_path: Path,
    duration: float = 5.0,
    fps: int = 15,
    viewport_height: int = 480,
) -> None:
    """Create a slow vertical pan video that reveals full PPE from head to boots."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    height, width = image.shape[:2]
    frame_count = int(duration * fps)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, viewport_height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video writer: {output_path}")

    for i in range(frame_count):
        t = i / max(1, frame_count - 1)
        y1 = int((height - viewport_height) * t)
        y2 = y1 + viewport_height
        writer.write(image[y1:y2, 0:width])

    writer.release()
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Generated video is empty: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "generated_demo_assets",
        help="Generated artifact directory (default: outputs/generated_demo_assets).",
    )
    args = parser.parse_args()
    if not DEMO_IMAGE.exists():
        raise FileNotFoundError(
            f"Demo image not found: {DEMO_IMAGE}. "
            "Please download CC0 sample images into examples/sample_images/ first."
        )

    output_root = args.output
    videos_dir = output_root / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    # Write matching localized zone configuration.
    image = cv2.imread(str(DEMO_IMAGE))
    height, width = image.shape[:2]
    zones = _make_construction_zones(width, height)
    zones_path = output_root / "sample_zones.yaml"
    zones_path.write_text(yaml.safe_dump(zones), encoding="utf-8")
    print(f"Wrote zone config to {zones_path}")

    # Generate a short surveillance-style panning clip.
    video_path = videos_dir / "construction_site_pan.mp4"
    _create_ken_burns_video(DEMO_IMAGE, video_path)
    print(f"Wrote demo video to {video_path}")

    # Generate a full-PPE vertical pan clip if the reference image is available.
    ppe_image = SAMPLE_IMAGES / "construction_site_ppe_01.jpg"
    if ppe_image.exists():
        ppe_video_path = videos_dir / "construction_ppe_pan.mp4"
        _create_vertical_pan_video(ppe_image, ppe_video_path)
        print(f"Wrote full-PPE demo video to {ppe_video_path}")


if __name__ == "__main__":
    main()
