"""Safety summary terminology tests."""

from utility_safety_ai.events.event import SafetyEvent
from utility_safety_ai.events.summary import aggregate_events


def _event(track_id: int | None) -> SafetyEvent:
    return SafetyEvent(
        event_id=f"event-{track_id}",
        timestamp="2026-07-13T00:00:00+00:00",
        source_type="video",
        source_path="demo.mp4",
        frame_index=0,
        time_seconds=0.0,
        risk_level="high",
        event_type="zone_intrusion",
        description="test",
        person_track_id=track_id,
        bbox=(0, 0, 10, 20),
        zone_id="zone",
        zone_name="Zone",
        snapshot_path=None,
        metadata={},
    )


def test_summary_counts_temporary_track_ids_not_people():
    summary = aggregate_events([_event(0), _event(0), _event(2), _event(None)])

    assert summary["unique_track_ids"] == 2
    assert "unique_persons" not in summary
