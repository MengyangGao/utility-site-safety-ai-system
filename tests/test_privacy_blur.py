"""Tests for privacy-preserving face/person blurring."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from utility_safety_ai.events.event import Detection
from utility_safety_ai.pipelines.image_pipeline import run_image_pipeline
from utility_safety_ai.privacy.face_blur import blur_faces


class _PrivacyDetector:
    conf = 0.25
    iou = 0.45
    device = "cpu"
    model_path = "privacy-test.pt"

    def __init__(self, detections: list[Detection]) -> None:
        self.detections = detections

    def reset_tracking(self) -> None:
        pass

    def predict(self, _image: np.ndarray) -> list[Detection]:
        return list(self.detections)


def test_blur_changes_upper_body_pixels():
    image = np.random.default_rng(1).integers(
        0, 255, (100, 100, 3), dtype=np.uint8
    )
    original = image.copy()
    detections = [
        Detection(
            class_id=0,
            class_name="person",
            confidence=0.9,
            bbox=(30, 20, 70, 90),
        )
    ]
    blurred = blur_faces(image, detections, enabled=True)

    assert blurred is image
    # Upper region inside the person bbox should be blurred.
    assert not np.array_equal(blurred[20:35, 30:70], original[20:35, 30:70])
    # Areas far from the person should remain unchanged.
    assert np.array_equal(blurred[80:95, 5:25], original[80:95, 5:25])


def test_blur_disabled_returns_original():
    image = np.random.default_rng(2).integers(
        0, 255, (50, 50, 3), dtype=np.uint8
    )
    original = image.copy()
    detections = [
        Detection(
            class_id=0,
            class_name="person",
            confidence=0.9,
            bbox=(10, 10, 30, 40),
        )
    ]
    blurred = blur_faces(image, detections, enabled=False)
    assert blurred is image
    assert np.array_equal(blurred, original)


@pytest.mark.parametrize("mode", ["gaussian", "pixelate", "solid"])
def test_supported_privacy_modes_redact_explicit_face(mode):
    image = np.random.default_rng(22).integers(0, 255, (80, 80, 3), dtype=np.uint8)
    original = image.copy()

    blur_faces(
        image,
        [Detection(1, "face", 0.9, (20, 20, 50, 50))],
        mode=mode,
    )

    assert not np.array_equal(image[20:50, 20:50], original[20:50, 20:50])


def test_invalid_privacy_mode_is_rejected():
    with pytest.raises(ValueError, match="privacy mode"):
        blur_faces(np.zeros((20, 20, 3), dtype=np.uint8), [], mode="identity")


def test_associated_face_uses_precise_box_without_upper_body_fallback():
    image = np.random.default_rng(3).integers(
        0, 255, (100, 100, 3), dtype=np.uint8
    )
    original = image.copy()
    detections = [
        Detection(0, "person", 0.95, (10, 10, 90, 90)),
        Detection(1, "face", 0.90, (40, 20, 60, 40)),
    ]

    blur_faces(image, detections)

    assert not np.array_equal(image[20:40, 40:60], original[20:40, 40:60])
    # This lies in the old broad fallback region but outside the precise face.
    assert np.array_equal(image[12:30, 15:35], original[12:30, 15:35])


def test_only_person_without_associated_face_uses_fallback():
    image = np.random.default_rng(4).integers(
        0, 255, (100, 120, 3), dtype=np.uint8
    )
    original = image.copy()
    detections = [
        Detection(0, "person", 0.95, (5, 10, 50, 90)),
        Detection(0, "person", 0.95, (65, 10, 110, 90)),
        Detection(1, "face", 0.90, (18, 18, 35, 38)),
    ]

    blur_faces(image, detections)

    assert np.array_equal(image[12:17, 8:16], original[12:17, 8:16])
    assert not np.array_equal(image[15:30, 75:100], original[15:30, 75:100])


def test_unassociated_face_and_person_fallback_are_both_blurred():
    image = np.random.default_rng(5).integers(
        0, 255, (100, 120, 3), dtype=np.uint8
    )
    original = image.copy()
    detections = [
        Detection(0, "person", 0.95, (5, 10, 50, 90)),
        Detection(1, "face", 0.90, (85, 20, 105, 40)),
    ]

    blur_faces(image, detections)

    assert not np.array_equal(image[15:30, 10:40], original[15:30, 10:40])
    assert not np.array_equal(image[20:40, 85:105], original[20:40, 85:105])


def test_event_snapshot_is_saved_from_privacy_processed_frame(tmp_path):
    source = tmp_path / "privacy-source.png"
    original = np.random.default_rng(6).integers(
        0, 255, (100, 100, 3), dtype=np.uint8
    )
    assert cv2.imwrite(str(source), original)
    detector = _PrivacyDetector(
        [
            Detection(0, "person", 0.95, (10, 10, 90, 90)),
            Detection(1, "face", 0.95, (40, 20, 60, 40)),
            Detection(2, "no_helmet", 0.90, (35, 15, 65, 45)),
        ]
    )

    _, events = run_image_pipeline(
        source,
        tmp_path / "outputs",
        detector,
        zones=[],
        blur_faces_enabled=True,
        run_id="privacy-run",
    )

    assert len(events) == 1
    snapshot = cv2.imread(
        str(
            tmp_path
            / "outputs"
            / "runs"
            / "privacy-run"
            / str(events[0].snapshot_path)
        )
    )
    assert snapshot is not None
    # Snapshot is the person crop (10:90, 10:90); translate the exact face ROI.
    original_face = original[20:40, 40:60]
    snapshot_face = snapshot[10:30, 30:50]
    original_detail = cv2.Laplacian(original_face, cv2.CV_64F).var()
    snapshot_detail = cv2.Laplacian(snapshot_face, cv2.CV_64F).var()
    assert snapshot_detail < original_detail * 0.2
