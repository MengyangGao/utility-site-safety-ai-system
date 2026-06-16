"""Tests for privacy-preserving face/person blurring."""

from __future__ import annotations

import numpy as np

from utility_safety_ai.events.event import Detection
from utility_safety_ai.privacy.face_blur import blur_faces


def test_blur_changes_upper_body_pixels():
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    original = image.copy()
    detections = [Detection(class_id=0, class_name="person", confidence=0.9, bbox=(30, 20, 70, 90))]
    blurred = blur_faces(image, detections, enabled=True)

    # Upper region inside the person bbox should be blurred.
    assert not np.array_equal(blurred[20:35, 30:70], original[20:35, 30:70])
    # Areas far from the person should remain unchanged.
    assert np.array_equal(blurred[80:95, 5:25], original[80:95, 5:25])


def test_blur_disabled_returns_original():
    image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    original = image.copy()
    detections = [Detection(class_id=0, class_name="person", confidence=0.9, bbox=(10, 10, 30, 40))]
    blurred = blur_faces(image, detections, enabled=False)
    assert np.array_equal(blurred, original)
