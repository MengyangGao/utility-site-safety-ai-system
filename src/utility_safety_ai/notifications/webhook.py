"""Timeout-bounded, signed webhook delivery with no redirects or raw media."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from math import isfinite
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from ..events.event import SafetyEvent
from ..events.store import public_event
from ..events.summary import aggregate_records


class DeliveryError(RuntimeError):
    def __init__(self, reason: str, *, permanent: bool = False):
        super().__init__(reason)
        self.permanent = permanent


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise HTTPError(req.full_url, code, "redirect refused", headers, fp)


def urlopen(request: Request, *, timeout: float):
    return build_opener(_NoRedirect).open(request, timeout=timeout)


def event_envelope(records: list[dict]) -> dict:
    return {
        "schema_version": "1.1",
        "summary": aggregate_records(records),
        "events": records,
    }


class WebhookSender:
    def __init__(self, url: str, *, secret: str = "", token: str = "", timeout_seconds: float = 5):
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Webhook URL must be an absolute HTTP(S) URL")
        if parsed.username or parsed.password:
            raise ValueError("Webhook credentials must be supplied separately, not in the URL")
        if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Non-loopback webhooks require HTTPS")
        if not isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("Webhook timeout must be finite and positive")
        self.url, self.secret, self.token, self.timeout = url, secret, token, timeout_seconds

    def __call__(self, record: dict) -> int:
        return self.send([record])

    def send(self, records: list[dict]) -> int:
        body = json.dumps(
            event_envelope(records), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        timestamp = str(int(time.time()))
        identity = hashlib.sha256(",".join(r["event_id"] for r in records).encode()).hexdigest()
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "utility-safety-ai/2.2",
            "Idempotency-Key": identity,
            "X-Safety-Timestamp": timestamp,
        }
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        if self.secret:
            signature = hmac.new(
                self.secret.encode(), timestamp.encode() + b"." + body, hashlib.sha256
            ).hexdigest()
            headers["X-Safety-Signature"] = "sha256=" + signature
        request = Request(self.url, data=body, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                status = int(response.status)
                if not 200 <= status < 300:
                    raise DeliveryError(
                        f"http_{status}", permanent=400 <= status < 500 and status not in {408, 429}
                    )
                return status
        except HTTPError as exc:
            raise DeliveryError(
                f"http_{exc.code}", permanent=300 <= exc.code < 500 and exc.code not in {408, 429}
            ) from None
        except (URLError, TimeoutError, OSError):
            raise DeliveryError("network_unavailable") from None


def deliver_webhook(
    events: list[SafetyEvent],
    url: str,
    *,
    timeout_seconds: float = 5,
    secret: str = "",
    token: str = "",
) -> int:
    """Compatibility entry point for an explicit event batch."""
    return WebhookSender(url, secret=secret, token=token, timeout_seconds=timeout_seconds).send(
        [public_event(e) for e in events]
    )
