"""Privacy-preserving face / upper-body blurring.

This module intentionally does **not** perform face recognition. It only
applies a Gaussian blur to detected face regions or, as a conservative
fallback, the upper portion of each person bounding box.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..events.event import Detection


def blur_faces(image: np.ndarray, detections: list[Detection], enabled: bool = True) -> np.ndarray:
    """Blur privacy-sensitive regions on a copy of the input image.

    Args:
        image: BGR image as a numpy array.
        detections: Normalized detections. ``person`` and ``face`` classes are
            blurred when ``enabled`` is True.
        enabled: Whether to apply blurring. If False, the original image is
            returned unchanged.

    Returns:
        The (possibly blurred) image.
    """
    if not enabled or image is None or image.size == 0:
        return image

    output = image.copy()

    for det in detections:
        if det.class_name == "face":
            _blur_region(output, det.bbox)
        elif det.class_name == "person":
            _blur_upper_body_fallback(output, det.bbox)

    return output


def _blur_region(image: np.ndarray, bbox: tuple[float, float, float, float]) -> None:
    x1, y1, x2, y2 = _clip_bbox(image, bbox)
    if x2 <= x1 or y2 <= y1:
        return
    roi = image[y1:y2, x1:x2]
    blurred = cv2.GaussianBlur(roi, (51, 51), 30)
    image[y1:y2, x1:x2] = blurred


def _blur_upper_body_fallback(
    image: np.ndarray, bbox: tuple[float, float, float, float]
) -> None:
    """Blur the upper ~30% of a person bounding box as a face-region fallback."""
    x1, y1, x2, y2 = bbox
    height = y2 - y1
    if height <= 0:
        return
    face_y2 = min(y1 + height * 0.35, y2)
    # Expand horizontally a little to cover shoulders/head.
    width = x2 - x1
    face_x1 = max(0, x1 - width * 0.05)
    face_x2 = min(image.shape[1], x2 + width * 0.05)
    _blur_region(image, (face_x1, y1, face_x2, face_y2))


def _clip_bbox(
    image: np.ndarray, bbox: tuple[float, float, float, float]
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    x1 = max(0, int(round(x1)))
    y1 = max(0, int(round(y1)))
    x2 = min(w, int(round(x2)))
    y2 = min(h, int(round(y2)))
    return x1, y1, x2, y2
