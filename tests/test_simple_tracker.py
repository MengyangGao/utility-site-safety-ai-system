"""Tests for fallback tracker lifecycle behavior."""

from __future__ import annotations

from utility_safety_ai.events.event import Detection
from utility_safety_ai.tracking.simple_tracker import SimpleTracker


def _person(bbox=(0, 0, 20, 40), track_id=None):
    return Detection(0, "person", 0.9, bbox, track_id)


def test_empty_frames_age_out_tracks():
    tracker = SimpleTracker(max_age=2)
    tracker.update([_person()])
    assert tracker.active_track_count == 1

    tracker.advance()
    tracker.update([])
    assert tracker.active_track_count == 1

    tracker.advance()
    assert tracker.active_track_count == 0


def test_reset_starts_ids_from_one_for_each_run():
    tracker = SimpleTracker()
    assert tracker.update([_person()])[0].track_id == 1
    tracker.reset()
    assert tracker.update([_person()])[0].track_id == 1


def test_upstream_track_id_is_preserved():
    tracker = SimpleTracker()
    tracked = tracker.update([_person(track_id=42)])
    assert tracked[0].track_id == 42


def test_motion_prediction_keeps_id_after_brief_miss():
    tracker = SimpleTracker(iou_threshold=0.22, max_age=3)
    first = tracker.update([_person((0, 0, 20, 40))])
    second = tracker.update([_person((8, 0, 28, 40))])
    tracker.advance()
    recovered = tracker.update([_person((24, 0, 44, 40))])

    assert first[0].track_id == second[0].track_id == recovered[0].track_id


def test_global_matching_keeps_neighbouring_tracks_distinct():
    tracker = SimpleTracker(iou_threshold=0.15)
    first = tracker.update(
        [_person((0, 0, 30, 60)), _person((40, 0, 70, 60))]
    )
    moved = tracker.update(
        [_person((7, 0, 37, 60)), _person((33, 0, 63, 60))]
    )

    assert [item.track_id for item in first] == [1, 2]
    assert [item.track_id for item in moved] == [1, 2]
