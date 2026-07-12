"""YOLO fine-tuning wrapper for custom PPE datasets."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ultralytics import YOLO

from ..detection.yolo_detector import _auto_device

logger = logging.getLogger(__name__)


def train(
    data: str | Path,
    model: str = "yolo11n.pt",
    epochs: int = 30,
    imgsz: int = 640,
    batch: int = 16,
    device: str | None = None,
    project: str | Path = "runs/train",
    name: str = "ppe",
    **kwargs,
) -> None:
    """Fine-tune a YOLO model on a custom dataset.

    Args:
        data: Path to the YOLO dataset YAML file.
        model: Base model name or path.
        epochs: Number of training epochs.
        imgsz: Input image size.
        batch: Batch size.
        device: Device string (``cpu``, ``mps``, ``0``, etc.). Auto-detected if None.
        project: Training run root directory.
        name: Run name.
        **kwargs: Extra arguments forwarded to ``YOLO.train``.
    """
    data_path = Path(data)
    if data_path.exists():
        data = str(data_path)
    # Otherwise treat it as an Ultralytics dataset name (e.g. "construction-ppe.yaml").

    if device is None:
        device = _auto_device()

    if kwargs.get("resume") and not Path(model).expanduser().is_file():
        raise FileNotFoundError(
            "--resume requires --model to point to an existing interrupted-run checkpoint"
        )

    logger.info("Starting YOLO training: model=%s data=%s epochs=%s device=%s", model, data, epochs, device)
    yolo = YOLO(model)
    if kwargs.get("resume"):
        checkpoint = getattr(yolo, "ckpt", None)
        resumable = (
            isinstance(checkpoint, dict)
            and isinstance(checkpoint.get("epoch"), int)
            and checkpoint["epoch"] >= 0
            and checkpoint.get("optimizer") is not None
        )
        if not resumable:
            raise ValueError(
                "--resume requires an unstripped interrupted-run checkpoint with "
                "epoch and optimizer state; completed best.pt/last.pt exports cannot resume"
            )
    yolo.train(
        data=str(data),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(project),
        name=name,
        **kwargs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a YOLO model for PPE detection.")
    parser.add_argument("--data", required=True, help="Path to dataset YAML.")
    parser.add_argument("--model", default="yolo11n.pt", help="Base model name or path.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="ppe")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume state from the existing checkpoint supplied with --model.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    train(
        data=args.data,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
