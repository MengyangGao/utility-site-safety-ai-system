"""Ultralytics YOLO wrapper that normalises outputs to :class:`Detection`."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from ..events.event import Detection
from .model_loader import resolve_model_path
from .runtime import load_yolo

logger = logging.getLogger(__name__)

# Per-class confidence floors. The user-supplied global ``conf`` remains a
# minimum for every class; listed classes may require a stricter floor.
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


def _first_result(results: Iterable[Any]) -> Any | None:
    """Return the first Ultralytics result from either a list or stream."""
    return next(iter(results), None)


def _auto_device() -> str:
    """Pick the best available inference device without requiring CUDA."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
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
        self.device = device or _auto_device()
        self.conf = conf
        self.iou = iou
        self.class_conf = dict(DEFAULT_CLASS_CONF if class_conf is None else class_conf)
        thresholds = [self.conf, self.iou, *self.class_conf.values()]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 <= float(value) <= 1.0
            for value in thresholds
        ):
            raise ValueError("All confidence and IoU thresholds must be between 0 and 1")
        # The global confidence is a true lower bound for every class. Submit it
        # to Ultralytics, then apply stricter per-class floors below.
        self.inference_conf = float(self.conf)
        resolved = resolve_model_path(model_path)
        logger.info("Loading YOLO model from %s on device %s", resolved, self.device)
        self.model = load_yolo(
            resolved, allow_download=model_path is not None and not Path(resolved).is_file()
        )

    def predict(self, image: np.ndarray | str | Path) -> list[Detection]:
        """Run detection on a single image/frame."""
        results = self.model(
            image,
            conf=self.inference_conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )
        result = _first_result(results)
        return [] if result is None else self._to_detections(result)

    def track(self, image: np.ndarray | str | Path) -> list[Detection]:
        """Run tracking on a single frame.

        Falls back to plain detection if the model does not expose tracking IDs.
        """
        try:
            results = self.model.track(
                image,
                conf=self.inference_conf,
                iou=self.iou,
                device=self.device,
                persist=True,
                verbose=False,
            )
            result = _first_result(results)
            return [] if result is None else self._to_detections(result)
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.debug("Tracking failed (%s); falling back to detection.", exc)
            return self.predict(image)

    def reset_tracking(self) -> None:
        """Discard Ultralytics tracker state before starting a new run."""
        predictor = getattr(self.model, "predictor", None)
        trackers = getattr(predictor, "trackers", ()) if predictor is not None else ()
        for tracker in trackers or ():
            reset = getattr(tracker, "reset", None)
            if callable(reset):
                reset()
        # Ultralytics constructs a fresh predictor/tracker lazily on the next
        # call. This also handles versions whose tracker has no public reset().
        if hasattr(self.model, "predictor"):
            self.model.predictor = None

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

            threshold = max(self.conf, self.class_conf.get(class_name, self.conf))
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
        # Ultralytics has already applied NMS. A second low-IoU suppression pass
        # here can erase nearby workers or separate glove/boot detections.
        return raw
