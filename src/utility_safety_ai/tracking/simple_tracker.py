"""Simple IoU-based tracker used when Ultralytics tracking is unavailable."""

from __future__ import annotations

from collections.abc import Iterable

from ..events.event import Detection


def _iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class SimpleTracker:
    """Assign consistent track IDs to detections using IoU matching.

    This is a minimal fallback tracker. It does not re-identify people across
    occlusions; it simply keeps IDs stable frame-to-frame for overlapping boxes.
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 5) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self._tracks: dict[int, tuple[tuple[float, float, float, float], int]] = {}
        self._next_id = 1

    def update(self, detections: list[Detection]) -> list[Detection]:
        """Assign/update track IDs for the supplied detections."""
        new_tracks: dict[int, tuple[tuple[float, float, float, float], int]] = {}
        matched_detections: list[Detection] = []
        unmatched_detections = list(detections)

        # Greedy best-IoU matching between current tracks and new detections.
        track_ids = list(self._tracks.keys())
        used = set()

        for tid in track_ids:
            last_bbox, age = self._tracks[tid]
            best_iou = self.iou_threshold
            best_idx = -1
            for idx, det in enumerate(unmatched_detections):
                if idx in used:
                    continue
                score = _iou(last_bbox, det.bbox)
                if score > best_iou:
                    best_iou = score
                    best_idx = idx

            if best_idx >= 0:
                det = unmatched_detections[best_idx]
                new_tracks[tid] = (det.bbox, 0)
                matched_detections.append(
                    Detection(
                        class_id=det.class_id,
                        class_name=det.class_name,
                        confidence=det.confidence,
                        bbox=det.bbox,
                        track_id=tid,
                    )
                )
                used.add(best_idx)

        # Create new tracks for unmatched detections.
        for idx, det in enumerate(unmatched_detections):
            if idx in used:
                continue
            tid = self._next_id
            self._next_id += 1
            new_tracks[tid] = (det.bbox, 0)
            matched_detections.append(
                Detection(
                    class_id=det.class_id,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    bbox=det.bbox,
                    track_id=tid,
                )
            )

        # Keep old tracks alive briefly to handle missed detections.
        for tid, (bbox, age) in self._tracks.items():
            if tid in new_tracks:
                continue
            if age + 1 < self.max_age:
                new_tracks[tid] = (bbox, age + 1)

        self._tracks = new_tracks
        return matched_detections

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1


def track_detections(detections: Iterable[Detection], tracker: SimpleTracker) -> list[Detection]:
    """Convenience wrapper around a :class:`SimpleTracker` instance."""
    return tracker.update(list(detections))
