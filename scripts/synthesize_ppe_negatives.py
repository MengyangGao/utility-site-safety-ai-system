"""Synthesize negative PPE training samples from positive labels.

For every positive PPE annotation (e.g. ``helmet``), this script creates a
synthetic ``no_helmet`` example by inpainting the annotated region so the PPE
item disappears. The new image is written next to the original data and the
label is updated to the corresponding negative class.

This is intended as a lightweight data-augmentation helper when real
negative-class images are scarce. Synthetic data is not a replacement for
real-world missing-PPE images, but it can help the model learn the geometry
of the negative class.

Example:
    python scripts/synthesize_ppe_negatives.py \
        --data datasets/construction-ppe/data.yaml \
        --classes helmet vest goggles \
        --output datasets/construction-ppe-synthetic
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_POSITIVE_CLASSES = ["helmet", "vest", "gloves", "boots", "goggles"]


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def _resolve_dataset_root(data_yaml: Path, config: dict) -> Path:
    """Resolve the dataset root from the ``path`` field in data.yaml."""
    raw_path = config.get("path", "")
    if raw_path:
        candidate = Path(raw_path)
        if candidate.is_absolute():
            return candidate
        return (data_yaml.parent / candidate).resolve()
    return data_yaml.parent.resolve()


def _negative_class_name(positive: str) -> str:
    return f"no_{positive}"


def _ensure_negative_class(names: dict[int, str], positive: str) -> tuple[dict[int, str], int]:
    """Return updated names dict and the class id for the negative class."""
    neg_name = _negative_class_name(positive)
    name_to_id = {name: cid for cid, name in names.items()}
    if neg_name in name_to_id:
        return names, name_to_id[neg_name]
    new_id = max(names.keys(), default=-1) + 1
    names[new_id] = neg_name
    return names, new_id


def _copy_original_split(
    root: Path,
    split: str,
    out_root: Path,
    names: dict[int, str],
) -> None:
    """Copy images and labels for one split into the output dataset."""
    src_img_dir = root / "images" / split
    src_lbl_dir = root / "labels" / split
    dst_img_dir = out_root / "images" / split
    dst_lbl_dir = out_root / "labels" / split

    if not src_img_dir.exists():
        logger.warning("Source image directory not found: %s", src_img_dir)
        return

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    for img_path in src_img_dir.iterdir():
        if not img_path.is_file():
            continue
        shutil.copy2(img_path, dst_img_dir / img_path.name)
        lbl_path = src_lbl_dir / f"{img_path.stem}.txt"
        if lbl_path.exists():
            shutil.copy2(lbl_path, dst_lbl_dir / lbl_path.name)


def _yolo_bbox_to_pixels(
    cx: float, cy: float, w: float, h: float, img_w: int, img_h: int
) -> tuple[int, int, int, int]:
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return x1, y1, x2, y2


def _pixels_to_yolo_bbox(
    x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int
) -> tuple[float, float, float, float]:
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    cx = (x1 + w / 2) / img_w
    cy = (y1 + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return cx, cy, nw, nh


def _inpaint_bbox(image: np.ndarray, x1: int, y1: int, x2: int, y2: int, radius: int) -> np.ndarray:
    """Inpaint a rectangular region using surrounding texture."""
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[max(0, y1) : y2, max(0, x1) : x2] = 255
    return cv2.inpaint(image, mask, radius, cv2.INPAINT_TELEA)


def _synthesize_for_split(
    root: Path,
    split: str,
    out_root: Path,
    names: dict[int, str],
    target_classes: list[str],
    max_per_class: int | None,
    inpaint_radius: int,
) -> int:
    """Generate synthetic negative samples for one split. Return count created."""
    src_img_dir = root / "images" / split
    src_lbl_dir = root / "labels" / split
    dst_img_dir = out_root / "images" / split
    dst_lbl_dir = out_root / "labels" / split

    if not src_img_dir.exists():
        return 0

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    name_to_id = {name: cid for cid, name in names.items()}
    target_ids = {name_to_id[name] for name in target_classes if name in name_to_id}
    if len(target_ids) != len(target_classes):
        missing = [c for c in target_classes if c not in name_to_id]
        logger.warning("Target classes not found in dataset names: %s", missing)

    # Map each target id to its negative id.
    neg_id_for: dict[int, int] = {}
    for cls in target_classes:
        if cls not in name_to_id:
            continue
        _, neg_id = _ensure_negative_class(names, cls)
        neg_id_for[name_to_id[cls]] = neg_id

    created = 0
    per_class_counts: dict[int, int] = dict.fromkeys(target_ids, 0)

    for lbl_path in sorted(src_lbl_dir.glob("*.txt")):
        if not lbl_path.stat().st_size:
            continue
        lines = lbl_path.read_text().strip().splitlines()
        parsed: list[tuple[int, float, float, float, float]] = []
        target_indices: list[int] = []
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cid = int(parts[0])
            bbox = tuple(float(v) for v in parts[1:5])
            parsed.append((cid, *bbox))
            if cid in target_ids:
                target_indices.append(i)

        if not target_indices:
            continue

        img_path = src_img_dir / f"{lbl_path.stem}.jpg"
        if not img_path.exists():
            for ext in (".jpeg", ".png", ".bmp"):
                img_path = src_img_dir / f"{lbl_path.stem}{ext}"
                if img_path.exists():
                    break
        if not img_path.exists():
            logger.warning("Image not found for label %s", lbl_path)
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning("Could not read image %s", img_path)
            continue
        img_h, img_w = image.shape[:2]

        for idx in target_indices:
            cid, cx, cy, w, h = parsed[idx]
            if max_per_class is not None and per_class_counts[cid] >= max_per_class:
                continue

            x1, y1, x2, y2 = _yolo_bbox_to_pixels(cx, cy, w, h, img_w, img_h)
            # Slightly enlarge the region to remove any visible PPE edges.
            margin_x = int((x2 - x1) * 0.05)
            margin_y = int((y2 - y1) * 0.05)
            x1 -= margin_x
            x2 += margin_x
            y1 -= margin_y
            y2 += margin_y

            synth_image = _inpaint_bbox(image.copy(), x1, y1, x2, y2, inpaint_radius)

            neg_cid = neg_id_for[cid]
            new_label_lines = []
            for j, (orig_cid, ocx, ocy, ow, oh) in enumerate(parsed):
                out_cid = neg_cid if j == idx else orig_cid
                new_label_lines.append(f"{out_cid} {ocx:.6f} {ocy:.6f} {ow:.6f} {oh:.6f}")

            synth_name = f"{img_path.stem}_syn_{names[neg_cid]}_{per_class_counts[cid]}"
            synth_img_path = dst_img_dir / f"{synth_name}.jpg"
            synth_lbl_path = dst_lbl_dir / f"{synth_name}.txt"

            cv2.imwrite(str(synth_img_path), synth_image)
            synth_lbl_path.write_text("\n".join(new_label_lines) + "\n", encoding="utf-8")

            per_class_counts[cid] += 1
            created += 1

    return created


def synthesize(
    data_yaml: str | Path,
    output_root: str | Path,
    classes: list[str] | None = None,
    splits: list[str] | None = None,
    max_per_class: int | None = None,
    inpaint_radius: int = 3,
) -> Path:
    """Create a new dataset that includes synthetic negative PPE samples."""
    data_yaml = Path(data_yaml)
    output_root = Path(output_root)
    config = _load_yaml(data_yaml)
    root = _resolve_dataset_root(data_yaml, config)
    splits = splits or ["train", "val"]
    target_classes = classes or DEFAULT_POSITIVE_CLASSES

    names: dict[int, str] = {int(k): str(v) for k, v in config.get("names", {}).items()}
    if not names:
        raise ValueError("data.yaml must contain a 'names' dictionary mapping class id to name.")

    # Ensure negative classes exist in the names dictionary.
    for cls in target_classes:
        names, _ = _ensure_negative_class(names, cls)

    # Copy original data first.
    for split in splits:
        _copy_original_split(root, split, output_root, names)

    # Generate synthetic negative samples.
    total_created = 0
    for split in splits:
        created = _synthesize_for_split(
            root,
            split,
            output_root,
            names,
            target_classes,
            max_per_class,
            inpaint_radius,
        )
        logger.info("Created %d synthetic negative samples for split '%s'", created, split)
        total_created += created

    # Write updated data.yaml.
    new_config = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": "images/val" if (output_root / "images" / "val").exists() else "images/train",
        "names": dict(sorted(names.items())),
    }
    if "test" in config:
        new_config["test"] = config["test"]
    if "nc" in config:
        new_config["nc"] = len(names)

    out_yaml = output_root / "data.yaml"
    _save_yaml(out_yaml, new_config)

    logger.info(
        "Synthetic dataset ready at %s (%d new samples, %d classes)",
        output_root,
        total_created,
        len(names),
    )
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthesize no_* PPE training samples by inpainting positive PPE regions."
    )
    parser.add_argument("--data", required=True, help="Path to Ultralytics-style data.yaml.")
    parser.add_argument("--output", required=True, help="Output dataset directory.")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_POSITIVE_CLASSES,
        help="Positive classes to synthesize negatives for.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Dataset splits to process.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Maximum synthetic samples per class per split.",
    )
    parser.add_argument(
        "--inpaint-radius",
        type=int,
        default=3,
        help="OpenCV inpainting radius around the masked region.",
    )
    args = parser.parse_args()
    synthesize(
        data_yaml=args.data,
        output_root=args.output,
        classes=args.classes,
        splits=args.splits,
        max_per_class=args.max_per_class,
        inpaint_radius=args.inpaint_radius,
    )


if __name__ == "__main__":
    main()
