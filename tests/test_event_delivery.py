"""Durability, idempotency, leases, signatures and authenticated review contracts."""

import hashlib
import hmac
import json
from contextlib import nullcontext
from dataclasses import replace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from test_webhook import _event

from utility_safety_ai.api import create_app
from utility_safety_ai.events.store import EventStore, public_event
from utility_safety_ai.notifications.delivery import DeliveryWorker, EventHub, MQTTSender
from utility_safety_ai.notifications.webhook import DeliveryError, WebhookSender


def test_event_and_outbox_are_idempotent_and_private(tmp_path):
    store = EventStore(tmp_path / "events.sqlite3")
    event = replace(
        _event(),
        source_path="rtsp://user:secret@camera/live",
        metadata={"run_id": "run1", "stream_segment": 2, "api_key": "secret"},
    )
    store.append(event, ("webhook",))
    store.append(event, ("webhook",))
    rows = store.list_events()
    assert len(rows) == 1
    assert rows[0]["source_path"] is None
    assert "secret" not in json.dumps(rows)
    assert rows[0]["metadata"]["stream_segment"] == 2
    assert store.delivery_summary()["pending"] == 1
    with pytest.raises(ValueError, match="reused"):
        store.append(replace(event, description="changed evidence"))


def test_expired_lease_can_be_recovered_and_old_owner_cannot_ack(tmp_path):
    store = EventStore(tmp_path / "events.sqlite3")
    store.append(_event(), ("webhook",))
    first = store.claim(("webhook",), now=100, lease_seconds=10)
    assert first and first["attempts"] == 1
    assert store.claim(("webhook",), now=105) is None
    restarted = EventStore(store.path)
    second = restarted.claim(("webhook",), now=111)
    assert second and second["attempts"] == 2
    assert not store.finish(first, now=112)
    assert restarted.finish(second, now=112)
    assert store.delivery_summary()["delivered"] == 1


def test_retry_backoff_dead_letter_and_explicit_retry(tmp_path):
    store = EventStore(tmp_path / "events.sqlite3")
    store.append(_event(), ("webhook",))
    now = 100.0
    for attempt in range(1, 7):
        item = store.claim(("webhook",), now=now)
        assert item["attempts"] == attempt
        store.finish(item, error="http_503", now=now)
        assert store.claim(("webhook",), now=now) is None
        now += 70
    assert store.delivery_summary()["dead"] == 1
    assert store.retry_failed() == 1
    assert store.claim(("webhook",), now=now)["attempts"] == 1
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM delivery_attempts").fetchone()[0] == 6


def test_sender_failure_does_not_lose_event_and_permanent_failure_stops_retries(tmp_path):
    store = EventStore(tmp_path / "events.sqlite3")
    store.append(_event(), ("webhook",))

    def fail(_):
        raise DeliveryError("http_401", permanent=True)

    worker = DeliveryWorker(store, {"webhook": fail})
    assert worker.run_once()
    assert store.delivery_summary()["dead"] == 1
    assert len(store.list_events()) == 1
    assert not worker.run_once()


def test_live_hub_delivers_without_waiting_for_pipeline_end(tmp_path):
    import threading

    delivered = threading.Event()

    def send(record):
        assert record["event_id"] == "e1"
        delivered.set()

    with EventHub(tmp_path, {"webhook": send}) as hub:
        hub.publish(_event())
        assert delivered.wait(2), "Event was delayed until after hub shutdown"
    assert hub.store.delivery_summary()["delivered"] == 1


def test_webhook_hmac_and_stable_idempotency(monkeypatch):
    from utility_safety_ai.notifications import webhook

    opener = Mock(return_value=nullcontext(Mock(status=202)))
    monkeypatch.setattr(webhook, "urlopen", opener)
    sender = WebhookSender(
        "https://receiver.example/hook", secret="test-secret", token="test-token"
    )
    record = public_event(_event())
    sender(record)
    request = opener.call_args.args[0]
    timestamp = request.get_header("X-safety-timestamp")
    expected = hmac.new(
        b"test-secret", timestamp.encode() + b"." + request.data, hashlib.sha256
    ).hexdigest()
    assert request.get_header("X-safety-signature") == "sha256=" + expected
    assert request.get_header("Authorization") == "Bearer test-token"
    key = request.get_header("Idempotency-key")
    sender(record)
    assert opener.call_args.args[0].get_header("Idempotency-key") == key


def test_webhook_redirect_is_refused_and_errors_do_not_leak_url(monkeypatch):
    from urllib.error import HTTPError

    from utility_safety_ai.notifications import webhook

    def redirect(*args, **kwargs):
        raise HTTPError("https://host/hook?secret=abc", 302, "private details", {}, None)

    monkeypatch.setattr(webhook, "urlopen", redirect)
    with pytest.raises(DeliveryError) as exc:
        WebhookSender("https://host/hook?secret=abc")(public_event(_event()))
    assert str(exc.value) == "http_302"
    assert exc.value.permanent


def test_reviews_append_without_modifying_evidence_and_api_is_protected(tmp_path):
    token = "test-token-with-at-least-24-characters"
    store = EventStore(tmp_path / "events.sqlite3")
    event = replace(_event(), metadata={"run_id": "r1"})
    store.append(event)
    snapshot = tmp_path / "runs/r1/snapshots/private.jpg"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(b"original evidence")
    with TestClient(create_app(tmp_path, token)) as client:
        assert client.get("/events").status_code == 401
        assert client.get("/events", headers={"Authorization": "Bearer bad"}).status_code == 401
        client.headers.update({"Authorization": "Bearer " + token})
        first = client.get("/events").json()
        assert first["next_cursor"] == 1
        assert client.get("/events?after=1").json()["events"] == []
        assert (
            client.post(
                "/events/e1/reviews",
                json={
                    "status": "confirmed",
                    "note": "Checked against footage.",
                    "reviewer": "Operator A",
                },
            ).status_code
            == 201
        )
        assert (
            client.post(
                "/events/e1/reviews",
                json={
                    "status": "false_positive",
                    "note": "Corrected after review.",
                    "reviewer": "Operator A",
                },
            ).status_code
            == 201
        )
        assert len(client.get("/events/e1/reviews").json()) == 2
        assert client.get("/events").json()["events"][0]["review_status"] == "false_positive"
        assert client.get("/events/e1/evidence").content == b"original evidence"
        assert client.get("/events/missing/evidence").status_code == 404
        assert (
            client.post(
                "/events/no/reviews", json={"status": "confirmed", "reviewer": "A"}
            ).status_code
            == 404
        )
        assert (
            client.post(
                "/events/e1/reviews", json={"status": "arbitrary", "reviewer": "A"}
            ).status_code
            == 422
        )
        assert client.get("/events?limit=999").status_code == 422
    assert snapshot.read_bytes() == b"original evidence"


def test_evidence_path_cannot_escape_run(tmp_path):
    store = EventStore(tmp_path / "events.sqlite3")
    store.append(replace(_event(), snapshot_path="../../private.jpg", metadata={"run_id": "r1"}))
    with pytest.raises(ValueError, match="escapes"):
        store.evidence_path("e1", tmp_path)


def test_mqtt_needs_acknowledgement(monkeypatch):
    import paho.mqtt.client as mqtt

    client = Mock()
    info = Mock()
    info.is_published.return_value = False
    client.publish.return_value = info
    monkeypatch.setattr(mqtt, "Client", Mock(return_value=client))
    sender = MQTTSender("localhost", 1883, tls=False, timeout_seconds=0.1)
    with pytest.raises(DeliveryError, match="ack_timeout"):
        sender(public_event(_event()))
    assert client.publish.call_args.kwargs["qos"] == 1
    assert client.publish.call_args.kwargs["retain"] is False
    client.disconnect.assert_called_once()
    info.is_published.return_value = True
    sender(public_event(_event()))


def test_insecure_remote_integrations_are_rejected():
    with pytest.raises(ValueError, match="HTTPS"):
        WebhookSender("http://remote.example/hook")
    with pytest.raises(ValueError, match="TLS"):
        MQTTSender("remote.example", 1883, tls=False)
    with pytest.raises(ValueError, match="token"):
        create_app(token="short")
