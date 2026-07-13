"""Privacy-preserving face / upper-body blurring.

This module intentionally does **not** perform face recognition. It redacts
explicit face detections, tries OpenCV's local Haar face detector inside person
boxes, and only then falls back to a compact head-region estimate.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..events.event import Detection

PRIVACY_MODES = ("gaussian", "pixelate", "solid")
_FACE_CASCADE: cv2.CascadeClassifier | None = None


def blur_faces(
    image: np.ndarray,
    detections: list[Detection],
    enabled: bool = True,
    mode: str = "gaussian",
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
    if mode not in PRIVACY_MODES:
        raise ValueError(f"Unsupported privacy mode: {mode}")
    if not enabled or image is None or image.size == 0:
        return image

    faces = [detection for detection in detections if detection.class_name == "face"]
    people = [
        detection for detection in detections if detection.class_name == "person"
    ]
    people_with_faces: set[int] = set()

    for face in faces:
        _redact_region(image, face.bbox, mode)
        person_index = _associated_person_index(face.bbox, people)
        if person_index is not None:
            people_with_faces.add(person_index)

    for person_index, person in enumerate(people):
        if person_index not in people_with_faces:
            detected_faces = _detect_faces_in_person(image, person.bbox)
            if detected_faces:
                for face_bbox in detected_faces:
                    _redact_region(image, face_bbox, mode)
            else:
                _blur_upper_body_fallback(image, person.bbox, mode)

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


def _redact_region(
    image: np.ndarray,
    bbox: tuple[float, float, float, float],
    mode: str,
) -> None:
    x1, y1, x2, y2 = _clip_bbox(image, bbox)
    if x2 <= x1 or y2 <= y1:
        return
    roi = image[y1:y2, x1:x2]
    if mode == "solid":
        image[y1:y2, x1:x2] = (15, 23, 42)
    elif mode == "pixelate":
        small_width = max(1, min(12, roi.shape[1] // 8))
        small_height = max(1, min(12, roi.shape[0] // 8))
        small = cv2.resize(roi, (small_width, small_height), interpolation=cv2.INTER_LINEAR)
        image[y1:y2, x1:x2] = cv2.resize(
            small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST
        )
    else:
        kernel = max(9, min(51, (min(roi.shape[:2]) // 2) * 2 + 1))
        image[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (kernel, kernel), 30)


def _blur_region(image: np.ndarray, bbox: tuple[float, float, float, float]) -> None:
    """Backward-compatible Gaussian redaction helper."""

    _redact_region(image, bbox, "gaussian")


def _detect_faces_in_person(
    image: np.ndarray,
    bbox: tuple[float, float, float, float],
) -> list[tuple[float, float, float, float]]:
    """Detect face-like regions locally without recognition or identity storage."""

    global _FACE_CASCADE
    x1, y1, x2, y2 = _clip_bbox(image, bbox)
    person_height = y2 - y1
    person_width = x2 - x1
    if person_height < 40 or person_width < 20:
        return []
    search_y2 = min(y2, y1 + int(person_height * 0.5))
    roi = image[y1:search_y2, x1:x2]
    if roi.size == 0:
        return []
    if _FACE_CASCADE is None:
        cv2_data = getattr(cv2, "data", None)
        haar_root = getattr(cv2_data, "haarcascades", "")
        if not haar_root:
            return []
        cascade_path = haar_root + "haarcascade_frontalface_default.xml"
        _FACE_CASCADE = cv2.CascadeClassifier(cascade_path)
    if _FACE_CASCADE.empty():
        return []
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    min_side = max(18, min(person_width, person_height) // 10)
    faces = _FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.12,
        minNeighbors=5,
        minSize=(min_side, min_side),
    )
    return [
        (float(x1 + fx), float(y1 + fy), float(x1 + fx + fw), float(y1 + fy + fh))
        for fx, fy, fw, fh in faces
    ]


def _blur_upper_body_fallback(
    image: np.ndarray,
    bbox: tuple[float, float, float, float],
    mode: str = "gaussian",
) -> None:
    """Redact a compact head region when no face detector can localize a face."""
    x1, y1, x2, y2 = bbox
    height = y2 - y1
    if height <= 0:
        return
    face_y2 = min(y1 + height * 0.28, y2)
    width = x2 - x1
    face_x1 = max(0, x1 + width * 0.12)
    face_x2 = min(image.shape[1], x2 - width * 0.12)
    _redact_region(image, (face_x1, y1, face_x2, face_y2), mode)


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
