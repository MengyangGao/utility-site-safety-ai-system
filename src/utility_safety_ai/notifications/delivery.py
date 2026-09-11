"""A small outbox worker keeps network latency out of the inference loop."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path

from ..events.event import SafetyEvent
from ..events.store import EventStore
from .webhook import DeliveryError, WebhookSender

logger = logging.getLogger(__name__)


class MQTTSender:
    """QoS 1 delivery; broker acknowledgement is required before outbox completion."""

    def __init__(
        self,
        host: str,
        port: int = 8883,
        *,
        tls: bool = True,
        topic: str = "utility-safety/events",
        username: str = "",
        password: str = "",
        timeout_seconds: float = 5,
    ):
        if not host or not 1 <= port <= 65535 or not topic or "+" in topic or "#" in topic:
            raise ValueError("MQTT requires a host, valid port and a concrete topic")
        if not tls and host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Non-loopback MQTT connections require TLS")
        self.host, self.port, self.tls, self.topic = host, port, tls, topic
        self.username, self.password, self.timeout = username, password, timeout_seconds

    def __call__(self, record: dict) -> None:
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            raise DeliveryError("mqtt_extra_not_installed", permanent=True) from None
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        client.connect_timeout = self.timeout
        if self.tls:
            client.tls_set()
        if self.username:
            client.username_pw_set(self.username, self.password)
        try:
            client.connect(self.host, self.port, keepalive=30)
            client.loop_start()
            info = client.publish(
                self.topic, json.dumps(record, sort_keys=True, allow_nan=False), qos=1, retain=False
            )
            info.wait_for_publish(timeout=self.timeout)
            if not info.is_published():
                raise DeliveryError("mqtt_ack_timeout")
        except DeliveryError:
            raise
        except (OSError, ValueError, RuntimeError):
            raise DeliveryError("mqtt_unavailable") from None
        finally:
            client.disconnect()
            client.loop_stop()


def configured_senders(webhook_url: str | None = None) -> dict[str, Callable]:
    senders: dict[str, Callable] = {}
    url = webhook_url or os.getenv("UTILITY_SAFETY_WEBHOOK_URL", "")
    if url:
        senders["webhook"] = WebhookSender(
            url,
            secret=os.getenv("UTILITY_SAFETY_WEBHOOK_SECRET", ""),
            token=os.getenv("UTILITY_SAFETY_WEBHOOK_TOKEN", ""),
        )
    host = os.getenv("UTILITY_SAFETY_MQTT_HOST", "")
    if host:
        tls = os.getenv("UTILITY_SAFETY_MQTT_TLS", "1") != "0"
        senders["mqtt"] = MQTTSender(
            host,
            int(os.getenv("UTILITY_SAFETY_MQTT_PORT", "8883" if tls else "1883")),
            tls=tls,
            topic=os.getenv("UTILITY_SAFETY_MQTT_TOPIC", "utility-safety/events"),
            username=os.getenv("UTILITY_SAFETY_MQTT_USERNAME", ""),
            password=os.getenv("UTILITY_SAFETY_MQTT_PASSWORD", ""),
        )
    return senders


class DeliveryWorker:
    def __init__(self, store: EventStore, senders: dict[str, Callable]):
        self.store, self.senders = store, senders
        self.stop = threading.Event()
        self.wake = threading.Event()
        self.thread: threading.Thread | None = None

    def run_once(self) -> bool:
        item = self.store.claim(tuple(self.senders))
        if not item:
            return False
        try:
            self.senders[item["target"]](json.loads(item["payload"]))
        except DeliveryError as exc:
            self.store.finish(item, error=str(exc), permanent=exc.permanent)
        except Exception:
            self.store.finish(item, error="sender_failed")
        else:
            self.store.finish(item)
        return True

    def _run(self):
        while not self.stop.is_set():
            try:
                if self.run_once():
                    continue
            except Exception:
                logger.error(
                    "Delivery worker could not access its outbox; pending records remain on disk."
                )
            self.wake.wait(0.25)
            self.wake.clear()

    def start(self):
        if self.senders:
            self.thread = threading.Thread(target=self._run, name="safety-outbox", daemon=True)
            self.thread.start()

    def close(self, grace_seconds: float = 2):
        if not self.thread:
            return
        deadline = time.monotonic() + grace_seconds
        self.wake.set()
        while time.monotonic() < deadline:
            summary = self.store.delivery_summary()
            if not summary["pending"] and not summary["sending"]:
                break
            time.sleep(0.05)
        self.stop.set()
        self.wake.set()
        self.thread.join(timeout=6)


class EventHub:
    """Context-managed integration index shared with the optional local REST API."""

    def __init__(self, output_root: Path, senders: dict[str, Callable] | None = None):
        self.store = EventStore(Path(output_root) / "events.sqlite3")
        self.senders = senders or {}
        self.worker = DeliveryWorker(self.store, self.senders)

    def __enter__(self):
        self.worker.start()
        return self

    def publish(self, event: SafetyEvent):
        self.store.append(event, tuple(self.senders))
        self.worker.wake.set()

    def __exit__(self, exc_type, exc, traceback):
        self.worker.close()
