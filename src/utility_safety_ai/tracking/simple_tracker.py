"""Simple IoU-based tracker used when Ultralytics tracking is unavailable."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import hypot, isfinite

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


def _centre_distance_score(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Return scale-normalized centre proximity in the range 0..1."""
    ax, ay = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
    bx, by = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    scale = max(
        hypot(a[2] - a[0], a[3] - a[1]),
        hypot(b[2] - b[0], b[3] - b[1]),
        1.0,
    )
    return max(0.0, 1.0 - hypot(ax - bx, ay - by) / scale)


@dataclass
class _TrackState:
    bbox: tuple[float, float, float, float]
    previous_bbox: tuple[float, float, float, float] | None = None
    age: int = 0

    def predicted_bbox(self) -> tuple[float, float, float, float]:
        """Linearly extrapolate the last motion while a detection is missed."""
        if self.previous_bbox is None:
            return self.bbox
        dx = ((self.bbox[0] + self.bbox[2]) - (self.previous_bbox[0] + self.previous_bbox[2])) / 2.0
        dy = ((self.bbox[1] + self.bbox[3]) - (self.previous_bbox[1] + self.previous_bbox[3])) / 2.0
        multiplier = min(self.age + 1, 3)
        return (
            self.bbox[0] + dx * multiplier,
            self.bbox[1] + dy * multiplier,
            self.bbox[2] + dx * multiplier,
            self.bbox[3] + dy * multiplier,
        )


class SimpleTracker:
    """Assign consistent track IDs to detections using IoU matching.

    The fallback uses globally ranked motion-aware matches. It still does not
    perform identity re-identification, but remains stable across brief misses
    and moderate movement where frame-to-frame IoU alone would create a new ID.
    """

    def __init__(
        self,
        iou_threshold: float = 0.22,
        max_age: int = 12,
        centre_weight: float = 0.28,
    ) -> None:
        if (
            isinstance(iou_threshold, bool)
            or not isinstance(iou_threshold, (int, float))
            or not 0.0 <= iou_threshold <= 1.0
        ):
            raise ValueError("iou_threshold must be between 0 and 1")
        if isinstance(max_age, bool) or not isinstance(max_age, int) or max_age < 1:
            raise ValueError("max_age must be a positive integer")
        if (
            isinstance(centre_weight, bool)
            or not isinstance(centre_weight, (int, float))
            or not isfinite(float(centre_weight))
            or not 0.0 <= centre_weight <= 1.0
        ):
            raise ValueError("centre_weight must be between 0 and 1")
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.centre_weight = float(centre_weight)
        self._tracks: dict[int, _TrackState] = {}
        self._next_id = 1

    def update(self, detections: list[Detection]) -> list[Detection]:
        """Assign/update track IDs for person detections only.

        Non-person detections (PPE items, faces, etc.) are returned unchanged so
        they do not inherit a person's track ID and pollute compliance reports.
        """
        person_dets = [d for d in detections if d.class_name == "person"]
        non_person_dets = [d for d in detections if d.class_name != "person"]

        new_tracks: dict[int, _TrackState] = {}
        person_to_tid: dict[int, int] = {}
        used: set[int] = set()
        used_track_ids: set[int] = set()

        # Respect IDs supplied by an upstream tracker when a frame contains a
        # mix of tracked and untracked detections. The fallback fills only gaps.
        for idx, det in enumerate(person_dets):
            if det.track_id is None:
                continue
            tid = det.track_id
            previous = self._tracks.get(tid)
            new_tracks[tid] = _TrackState(
                bbox=det.bbox,
                previous_bbox=previous.bbox if previous is not None else None,
            )
            person_to_tid[idx] = tid
            used.add(idx)
            used_track_ids.add(tid)
            self._next_id = max(self._next_id, tid + 1)

        # Build all plausible pairs first, then claim them globally by score.
        # This avoids track iteration order stealing a detection from a closer
        # neighbouring person in crowded scenes.
        candidates: list[tuple[float, float, int, int]] = []
        for tid, state in self._tracks.items():
            if tid in used_track_ids:
                continue
            predicted = state.predicted_bbox()
            for idx, det in enumerate(person_dets):
                if idx in used:
                    continue
                overlap = max(_iou(state.bbox, det.bbox), _iou(predicted, det.bbox))
                centre = _centre_distance_score(predicted, det.bbox)
                score = (1.0 - self.centre_weight) * overlap + self.centre_weight * centre
                if overlap >= self.iou_threshold or score >= self.iou_threshold + 0.12:
                    candidates.append((score, overlap, tid, idx))
        candidates.sort(reverse=True)
        claimed_tracks = set(used_track_ids)
        for _score, _overlap, tid, idx in candidates:
            if tid in claimed_tracks or idx in used:
                continue
            state = self._tracks[tid]
            det = person_dets[idx]
            new_tracks[tid] = _TrackState(bbox=det.bbox, previous_bbox=state.bbox)
            person_to_tid[idx] = tid
            claimed_tracks.add(tid)
            used.add(idx)

        for tid, state in self._tracks.items():
            if tid in claimed_tracks:
                continue
            if state.age + 1 <= self.max_age:
                new_tracks[tid] = _TrackState(
                    bbox=state.bbox,
                    previous_bbox=state.previous_bbox,
                    age=state.age + 1,
                )

        # Create new tracks for unmatched person detections.
        for idx, det in enumerate(person_dets):
            if idx in used:
                continue
            tid = self._next_id
            self._next_id += 1
            new_tracks[tid] = _TrackState(bbox=det.bbox)
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
