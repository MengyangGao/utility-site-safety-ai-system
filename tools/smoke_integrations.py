"""Exercise real loopback webhook retries/HMAC and MQTT QoS 1 with synthetic events."""

import argparse
import hashlib
import hmac
import json
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import paho.mqtt.client as mqtt

from utility_safety_ai.events.event import SafetyEvent
from utility_safety_ai.events.store import public_event
from utility_safety_ai.notifications.delivery import EventHub, MQTTSender
from utility_safety_ai.notifications.webhook import WebhookSender


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/integration-smoke.json"))
    args = parser.parse_args()
    binary = shutil.which("mosquitto")
    if binary is None and Path("/opt/homebrew/opt/mosquitto/sbin/mosquitto").is_file():
        binary = "/opt/homebrew/opt/mosquitto/sbin/mosquitto"
    if binary is None:
        raise SystemExit("Install Mosquitto to run this optional loopback integration check.")
    receipts = []
    delivered = threading.Event()
    secret = b"local-test-secret"

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            timestamp = self.headers["X-Safety-Timestamp"]
            expected = hmac.new(
                secret, timestamp.encode() + b"." + body, hashlib.sha256
            ).hexdigest()
            valid = hmac.compare_digest(
                self.headers.get("X-Safety-Signature", ""), "sha256=" + expected
            )
            receipts.append(
                {
                    "valid_signature": valid,
                    "idempotency_key": self.headers["Idempotency-Key"],
                    "payload": json.loads(body),
                }
            )
            self.send_response(503 if len(receipts) == 1 else 202)
            self.end_headers()
            if len(receipts) > 1:
                delivered.set()

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    event = SafetyEvent(
        "integration-fixture",
        "2026-09-11T00:00:00Z",
        "image",
        "synthetic fixture",
        0,
        0.0,
        "low",
        "integration_check",
        "Synthetic connectivity check",
        None,
        None,
        None,
        None,
        None,
        {"run_id": "loopback-test"},
    )
    with tempfile.TemporaryDirectory() as temporary:
        try:
            with EventHub(
                Path(temporary),
                {
                    "webhook": WebhookSender(
                        f"http://127.0.0.1:{server.server_port}/events", secret=secret.decode()
                    )
                },
            ) as hub:
                hub.publish(event)
                assert delivered.wait(8), (
                    "Webhook retry did not arrive while the producer was running"
                )
            assert hub.store.delivery_summary()["delivered"] == 1
            assert len(receipts) == 2 and all(item["valid_signature"] for item in receipts)
            assert receipts[0]["idempotency_key"] == receipts[1]["idempotency_key"]
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            config = Path(temporary) / "mosquitto.conf"
            config.write_text(
                f"listener {port} 127.0.0.1\nallow_anonymous true\npersistence false\n"
            )
            with (Path(temporary) / "broker.log").open("w") as log:
                broker = subprocess.Popen([binary, "-c", str(config)], stdout=log, stderr=log)
                client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
                subscribed, received = threading.Event(), threading.Event()
                messages = []

                def on_connect(client, userdata, flags, reason, properties):
                    client.subscribe("utility-safety/events", qos=1)

                def on_subscribe(client, userdata, mid, reason_codes, properties):
                    subscribed.set()

                def on_message(client, userdata, message):
                    messages.append(json.loads(message.payload))
                    received.set()

                client.on_connect, client.on_subscribe, client.on_message = (
                    on_connect,
                    on_subscribe,
                    on_message,
                )
                try:
                    for _ in range(50):
                        try:
                            client.connect("127.0.0.1", port)
                            break
                        except OSError:
                            time.sleep(0.1)
                    client.loop_start()
                    assert subscribed.wait(3)
                    MQTTSender("127.0.0.1", port, tls=False)(public_event(event))
                    assert received.wait(3)
                    assert messages[0]["event_id"] == event.event_id
                finally:
                    client.disconnect()
                    client.loop_stop()
                    broker.terminate()
                    try:
                        broker.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        broker.kill()
                        broker.wait()
            result = {
                "result": "passed",
                "scope": "Real loopback HTTP and MQTT; synthetic event, no external endpoint or physical camera.",
                "webhook_requests": len(receipts),
                "webhook_retry_after_503": True,
                "hmac_verified": True,
                "stable_idempotency_key": True,
                "mqtt_qos": 1,
                "mqtt_subscriber_received": True,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    main()
