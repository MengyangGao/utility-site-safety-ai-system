"""Validate a trained PPE model and save per-class metrics, curves, and failure cases."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate(
    model_path: str | Path,
    output_dir: str | Path = "outputs/validation",
    data: str = "construction-ppe.yaml",
) -> dict:
    """Run COCO-style validation on a PPE dataset.

    Produces:
      - metrics.json with per-class precision/recall/mAP50/mAP50-95
      - confusion_matrix.png and confusion_matrix_normalized.png
      - PR_curve.png and F1_curve.png
      - Validation-batch label/prediction comparison images

    Args:
        model_path: Path to the trained model weights.
        output_dir: Directory where validation artifacts are saved.
        data: Ultralytics dataset YAML or dataset name.
    """
    import torch
    from ultralytics import YOLO
    from ultralytics import __version__ as ultralytics_version

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. "
            "Train a model first or place weights at models/ppe_yolo11n.pt"
        )

    model = YOLO(str(model_path))
    save_json = importlib.util.find_spec("pycocotools") is not None
    if not save_json:
        logger.info(
            "pycocotools is not installed; continuing without COCO JSON export. "
            "Metric computation and validation plots are unaffected."
        )
    metrics = model.val(
        data=data,
        verbose=False,
        plots=True,
        save=True,
        save_json=save_json,
    )

    # Build a serialisable per-class report.
    names = {int(k): v for k, v in metrics.names.items()}
    per_class = {}
    for i, class_name in names.items():
        per_class[class_name] = {
            "precision": round(float(metrics.box.p[i]), 4),
            "recall": round(float(metrics.box.r[i]), 4),
            "map50": round(float(metrics.box.ap50[i]), 4),
            "map50_95": round(float(metrics.box.ap[i]), 4),
        }

    report = {
        "schema_version": "1.0",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "model": str(model_path),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "data": data,
        "split": "val",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "ultralytics": ultralytics_version,
            "device": str(getattr(model, "device", "auto")),
        },
        "map50": round(float(metrics.box.map50), 4),
        "map50_95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "per_class": per_class,
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    # Copy Ultralytics validation plots and prediction images.
    val_dir = Path(metrics.save_dir)
    plots = {
        "confusion_matrix.png": "confusion_matrix.png",
        "confusion_matrix_normalized.png": "confusion_matrix_normalized.png",
        "BoxPR_curve.png": "PR_curve.png",
        "BoxF1_curve.png": "F1_curve.png",
        "BoxP_curve.png": "Precision_curve.png",
        "BoxR_curve.png": "Recall_curve.png",
    }
    for src_name, dst_name in plots.items():
        src = val_dir / src_name
        if src.exists():
            shutil.copy(src, output_dir / dst_name)
            logger.info("Copied %s", dst_name)

    # Copy validation batch comparison images (GT labels vs predictions).
    batch_dir = output_dir / "val_batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(val_dir.iterdir()):
        if src.name.startswith("val_batch") and src.suffix.lower() in (".jpg", ".jpeg", ".png"):
            shutil.copy(src, batch_dir / src.name)
            logger.info("Copied %s", src.name)

    # Copy a subset of annotated validation images as a prediction gallery.
    pred_dir = val_dir / "predictions"
    gallery_dir = output_dir / "predictions"
    gallery_dir.mkdir(parents=True, exist_ok=True)
    if pred_dir.exists():
        copied = 0
        for src in sorted(pred_dir.iterdir()):
            if src.suffix.lower() in (".jpg", ".jpeg", ".png"):
                shutil.copy(src, gallery_dir / src.name)
                copied += 1
                if copied >= 50:
                    break
        logger.info("Copied %d annotated validation predictions to %s", copied, gallery_dir)

    logger.info(
        "Validation complete. mAP50=%s, mAP50-95=%s",
        report["map50"],
        report["map50_95"],
    )
    logger.info("Results saved to %s", output_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a PPE YOLO model.")
    parser.add_argument("--model", default="models/ppe_yolo11n.pt", help="Path to model weights.")
    parser.add_argument("--output", default="outputs/validation", help="Directory for validation artifacts.")
    parser.add_argument(
        "--data",
        default="construction-ppe.yaml",
        help="Ultralytics dataset YAML or dataset name (default: construction-ppe.yaml).",
    )
    args = parser.parse_args()
    validate(args.model, args.output, data=args.data)


if __name__ == "__main__":
    main()
