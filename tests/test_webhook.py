"""Webhook adapter tests avoid real network access."""

import json
from contextlib import nullcontext
from unittest.mock import Mock

import pytest

from utility_safety_ai.events.event import SafetyEvent
from utility_safety_ai.notifications import webhook


def _event() -> SafetyEvent:
    return SafetyEvent(
        event_id="e1",
        timestamp="2026-01-01T00:00:00Z",
        source_type="image",
        source_path="demo.jpg",
        frame_index=0,
        time_seconds=0.0,
        risk_level="high",
        event_type="zone_intrusion",
        description="test",
        person_track_id=1,
        bbox=(1, 2, 3, 4),
        zone_id="z1",
        zone_name="Zone",
        snapshot_path="snapshots/private.jpg",
    )


def test_webhook_posts_json_without_local_snapshot_path(monkeypatch):
    response = Mock(status=202)
    opener = Mock(return_value=nullcontext(response))
    monkeypatch.setattr(webhook, "urlopen", opener)

    assert webhook.deliver_webhook([_event()], "https://alerts.example.test/hook") == 202
    request = opener.call_args.args[0]
    assert json.loads(request.data)["events"][0]["snapshot_path"] is None
    assert b"snapshots/private.jpg" not in request.data


def test_webhook_rejects_non_http_url():
    with pytest.raises(ValueError, match="HTTP"):
        webhook.deliver_webhook([_event()], "file:///tmp/report")
