"""Behavior and rendering-cost tests for frame annotation."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import ImageFont

from utility_safety_ai.events.event import Detection, SafetyEvent
from utility_safety_ai.visualization import annotator
from utility_safety_ai.zones.zone import Zone


def _event() -> SafetyEvent:
    return SafetyEvent(
        event_id="event-1",
        timestamp="2026-01-01T00:00:00Z",
        source_type="image",
        source_path="test.jpg",
        frame_index=None,
        time_seconds=None,
        risk_level="high",
        event_type="zone_intrusion",
        description="Zone intrusion",
        person_track_id=1,
        bbox=(10, 10, 40, 80),
        zone_id="zone",
        zone_name="Zone",
        snapshot_path=None,
        metadata={},
    )


def test_normalized_zone_is_scaled_to_frame_pixels():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    zone = Zone(
        id="normalized",
        name="Normalized",
        risk_level="high",
        polygon=[(0.5, 0.5), (0.9, 0.5), (0.9, 0.9), (0.5, 0.9)],
        coordinate_space="normalized",
    )

    annotator.draw_zones(image, [zone])

    assert np.any(image[70, 70])
    assert not np.any(image[20, 20])


def test_pil_labels_use_one_round_trip_color_conversion_per_frame(monkeypatch):
    image = np.zeros((120, 180, 3), dtype=np.uint8)
    detections = [
        Detection(0, "person", 0.95, (10, 20, 60, 100), track_id=1),
        Detection(1, "helmet", 0.90, (20, 25, 45, 45)),
    ]
    zone = Zone(
        id="zone",
        name="Restricted",
        risk_level="high",
        polygon=[(0, 70), (100, 70), (100, 115), (0, 115)],
    )
    monkeypatch.setattr(annotator, "_needs_pil", lambda _text: True)
    monkeypatch.setattr(
        annotator,
        "_load_pil_font",
        lambda _size: ImageFont.load_default(),
    )
    real_cvt_color = cv2.cvtColor
    conversions: list[int] = []

    def counted_cvt_color(array, code):
        conversions.append(code)
        return real_cvt_color(array, code)

    monkeypatch.setattr(annotator.cv2, "cvtColor", counted_cvt_color)

    result = annotator.annotate_image(image, [zone], detections, [_event()])

    assert result.shape == image.shape
    assert conversions == [cv2.COLOR_BGR2RGB, cv2.COLOR_RGB2BGR]


def test_annotate_image_preserves_input_frame():
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    original = image.copy()

    result = annotator.annotate_image(
        image,
        zones=[],
        detections=[Detection(0, "person", 0.9, (10, 10, 40, 70))],
        events=[],
    )

    assert np.array_equal(image, original)
    assert not np.array_equal(result, original)


def test_unrelated_general_model_classes_are_logged_but_not_annotated():
    image = np.zeros((80, 100, 3), dtype=np.uint8)

    annotator.draw_detections(
        image,
        [Detection(13, "bench", 0.9, (10, 10, 90, 70))],
    )

    assert not np.any(image)
