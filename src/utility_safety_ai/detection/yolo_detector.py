"""Ultralytics YOLO wrapper that normalises outputs to :class:`Detection`."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..events.event import Detection
from .model_loader import resolve_model_path

logger = logging.getLogger(__name__)

# Per-class confidence floors. A detection is kept when its confidence is
# at least the class-specific value below, or at least the global ``conf``
# passed to the detector for classes not listed.
DEFAULT_CLASS_CONF: dict[str, float] = {
    "person": 0.30,
    "helmet": 0.35,
    "vest": 0.30,
    "gloves": 0.15,
    "boots": 0.20,
    "goggles": 0.20,
    "no_helmet": 0.30,
    "no_vest": 0.30,
    "no_gloves": 0.15,
    "no_boots": 0.20,
    "no_goggles": 0.20,
    "no_goggle": 0.20,
}

# Classes that should never be emitted as safety-relevant detections.
IGNORE_CLASSES = {"none"}


def _auto_device() -> str:
    """Pick the best available inference device without requiring CUDA."""
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    except Exception:  # pragma: no cover
        pass
    return "cpu"


class YoloDetector:
    """Lightweight wrapper around an Ultralytics YOLO model."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str | None = None,
        conf: float = 0.25,
        iou: float = 0.45,
        class_conf: dict[str, float] | None = None,
    ) -> None:
        from ultralytics import YOLO

        self.device = device or _auto_device()
        self.conf = conf
        self.iou = iou
        self.class_conf = class_conf or DEFAULT_CLASS_CONF
        resolved = resolve_model_path(model_path)
        logger.info("Loading YOLO model from %s on device %s", resolved, self.device)
        self.model = YOLO(resolved)

    def predict(self, image: np.ndarray | str | Path) -> list[Detection]:
        """Run detection on a single image/frame."""
        results = self.model(
            image,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )
        return self._to_detections(results[0])

    def track(self, image: np.ndarray | str | Path) -> list[Detection]:
        """Run tracking on a single frame.

        Falls back to plain detection if the model does not expose tracking IDs.
        """
        try:
            results = self.model.track(
                image,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                persist=True,
                verbose=False,
            )
            return self._to_detections(results[0])
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.debug("Tracking failed (%s); falling back to detection.", exc)
            return self.predict(image)

    def _to_detections(self, result: Any) -> list[Detection]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        names = result.names
        raw: list[Detection] = []
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = map(float, xyxy)
            confidence = float(boxes.conf[i].cpu().item())
            class_id = int(boxes.cls[i].cpu().item())
            class_name = names.get(class_id, str(class_id)).lower()

            if class_name in IGNORE_CLASSES:
                continue

            threshold = self.class_conf.get(class_name, self.conf)
            if confidence < threshold:
                continue

            track_id = None
            if boxes.id is not None:
                track_id = int(boxes.id[i].cpu().item())
            raw.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                    track_id=track_id,
                )
            )
        return self._class_wise_nms(raw)

    def _class_wise_nms(
        self,
        detections: list[Detection],
        iou_threshold: float = 0.35,
    ) -> list[Detection]:
        """Suppress near-duplicate boxes of the same class after model NMS.

        Ultralytics NMS can still emit overlapping boxes for heavily-represented
        classes; this lightweight per-class cleanup keeps annotations tidy.
        """
        by_class: dict[str, list[Detection]] = {}
        for det in detections:
            by_class.setdefault(det.class_name, []).append(det)

        kept: list[Detection] = []
        for dets in by_class.values():
            dets = sorted(dets, key=lambda d: d.confidence, reverse=True)
            while dets:
                current = dets.pop(0)
                kept.append(current)
                dets = [d for d in dets if _iou(current.bbox, d.bbox) <= iou_threshold]
        return kept


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Compute intersection-over-union of two xyxy boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0
