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
        if (
            isinstance(iou_threshold, bool)
            or not isinstance(iou_threshold, (int, float))
            or not 0.0 <= iou_threshold <= 1.0
        ):
            raise ValueError("iou_threshold must be between 0 and 1")
        if isinstance(max_age, bool) or not isinstance(max_age, int) or max_age < 1:
            raise ValueError("max_age must be a positive integer")
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self._tracks: dict[int, tuple[tuple[float, float, float, float], int]] = {}
        self._next_id = 1

    def update(self, detections: list[Detection]) -> list[Detection]:
        """Assign/update track IDs for person detections only.

        Non-person detections (PPE items, faces, etc.) are returned unchanged so
        they do not inherit a person's track ID and pollute compliance reports.
        """
        person_dets = [d for d in detections if d.class_name == "person"]
        non_person_dets = [d for d in detections if d.class_name != "person"]

        new_tracks: dict[int, tuple[tuple[float, float, float, float], int]] = {}
        person_to_tid: dict[int, int] = {}
        used: set[int] = set()
        used_track_ids: set[int] = set()

        # Respect IDs supplied by an upstream tracker when a frame contains a
        # mix of tracked and untracked detections. The fallback fills only gaps.
        for idx, det in enumerate(person_dets):
            if det.track_id is None:
                continue
            tid = det.track_id
            new_tracks[tid] = (det.bbox, 0)
            person_to_tid[idx] = tid
            used.add(idx)
            used_track_ids.add(tid)
            self._next_id = max(self._next_id, tid + 1)

        # Greedy best-IoU matching between current tracks and new person detections.
        for tid, (last_bbox, age) in self._tracks.items():
            if tid in used_track_ids:
                continue
            best_iou = self.iou_threshold
            best_idx = -1
            for idx, det in enumerate(person_dets):
                if idx in used:
                    continue
                score = _iou(last_bbox, det.bbox)
                if score >= best_iou:
                    best_iou = score
                    best_idx = idx

            if best_idx >= 0:
                det = person_dets[best_idx]
                new_tracks[tid] = (det.bbox, 0)
                person_to_tid[best_idx] = tid
                used.add(best_idx)
            elif age + 1 <= self.max_age:
                # Keep the old track alive briefly for missed detections.
                new_tracks[tid] = (last_bbox, age + 1)

        # Create new tracks for unmatched person detections.
        for idx, det in enumerate(person_dets):
            if idx in used:
                continue
            tid = self._next_id
            self._next_id += 1
            new_tracks[tid] = (det.bbox, 0)
            person_to_tid[idx] = tid
            used.add(idx)

        self._tracks = new_tracks

        # Rebuild the detection list with stable track IDs for persons.
        matched_detections: list[Detection] = []
        for idx, det in enumerate(person_dets):
            matched_detections.append(
                Detection(
                    class_id=det.class_id,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    bbox=det.bbox,
                    track_id=person_to_tid.get(idx),
                )
            )
        matched_detections.extend(non_person_dets)
        return matched_detections

    def reset(self) -> None:
        """Start a new run with no IDs or retained tracks."""
        self._tracks.clear()
        self._next_id = 1

    def advance(self) -> None:
        """Age tracks by one empty frame.

        Pipelines should call this (or ``update([])``) when a detector returns no
        boxes; otherwise stale fallback tracks never expire during blank frames.
        """
        self.update([])

    @property
    def active_track_count(self) -> int:
        """Number of live tracks, including temporarily missed people."""
        return len(self._tracks)


def track_detections(detections: Iterable[Detection], tracker: SimpleTracker) -> list[Detection]:
    """Convenience wrapper around a :class:`SimpleTracker` instance."""
    return tracker.update(list(detections))
