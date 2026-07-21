"""Generate a demo video from the provenance-tracked Construction-PPE sample."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_IMAGE = REPO_ROOT / "examples" / "sample_images" / "construction_site_ppe_01.jpg"


def create_vertical_pan_video(
    image_path: Path,
    output_path: Path,
    *,
    duration: float = 5.0,
    fps: int = 15,
    viewport_height: int = 480,
) -> None:
    """Create a slow vertical pan that reveals PPE from head to boots."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    height, width = image.shape[:2]
    if viewport_height > height:
        raise ValueError("Viewport height cannot exceed the source image height")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, viewport_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video writer: {output_path}")

    frame_count = int(duration * fps)
    try:
        for index in range(frame_count):
            progress = index / max(1, frame_count - 1)
            y1 = int((height - viewport_height) * progress)
            writer.write(image[y1 : y1 + viewport_height, 0:width])
    finally:
        writer.release()

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Generated video is empty: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEMO_IMAGE)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "generated_demo_assets" / "construction_ppe_pan.mp4",
    )
    args = parser.parse_args()
    if not args.source.is_file():
        raise FileNotFoundError(f"Demo image not found: {args.source}")
    create_vertical_pan_video(args.source, args.output)
    print(f"Wrote demo video to {args.output}")


if __name__ == "__main__":
    main()
