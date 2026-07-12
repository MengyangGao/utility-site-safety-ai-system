"""Privacy-preserving face / upper-body blurring.

This module intentionally does **not** perform face recognition. It only
applies a Gaussian blur to detected face regions or, as a conservative
fallback, the upper portion of each person bounding box.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..events.event import Detection


def blur_faces(
    image: np.ndarray,
    detections: list[Detection],
    enabled: bool = True,
) -> np.ndarray:
    """Blur privacy-sensitive regions in-place.

    Pipelines pass a dedicated display-frame copy into this function, avoiding a
    second full-frame allocation for every video frame. When an explicit face is
    associated with a person, only that precise face box is blurred. People with
    no associated face use the conservative upper-body fallback.

    Args:
        image: BGR image as a numpy array.
        detections: Normalized detections. ``person`` and ``face`` classes are
            blurred when ``enabled`` is True.
        enabled: Whether to apply blurring. If False, the original image is
            returned unchanged.

    Returns:
        The same image object, possibly blurred.
    """
    if not enabled or image is None or image.size == 0:
        return image

    faces = [detection for detection in detections if detection.class_name == "face"]
    people = [
        detection for detection in detections if detection.class_name == "person"
    ]
    people_with_faces: set[int] = set()

    for face in faces:
        _blur_region(image, face.bbox)
        person_index = _associated_person_index(face.bbox, people)
        if person_index is not None:
            people_with_faces.add(person_index)

    for person_index, person in enumerate(people):
        if person_index not in people_with_faces:
            _blur_upper_body_fallback(image, person.bbox)

    return image


def _associated_person_index(
    face_bbox: tuple[float, float, float, float],
    people: list[Detection],
) -> int | None:
    """Return the best person for a face using containment and face coverage."""
    fx1, fy1, fx2, fy2 = face_bbox
    face_width = max(0.0, fx2 - fx1)
    face_height = max(0.0, fy2 - fy1)
    face_area = face_width * face_height
    if face_area <= 0:
        return None

    center_x = (fx1 + fx2) / 2.0
    center_y = (fy1 + fy2) / 2.0
    best_index: int | None = None
    best_score = 0.0
    for index, person in enumerate(people):
        px1, py1, px2, py2 = person.bbox
        center_inside = px1 <= center_x <= px2 and py1 <= center_y <= py2
        intersection = max(0.0, min(fx2, px2) - max(fx1, px1)) * max(
            0.0, min(fy2, py2) - max(fy1, py1)
        )
        face_coverage = intersection / face_area
        if not center_inside and face_coverage < 0.5:
            continue
        score = face_coverage + (1.0 if center_inside else 0.0)
        if score > best_score:
            best_score = score
            best_index = index
    return best_index


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
