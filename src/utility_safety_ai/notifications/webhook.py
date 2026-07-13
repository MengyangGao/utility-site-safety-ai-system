"""Minimal JSON webhook delivery for operator-owned integrations."""

from __future__ import annotations

import json
from dataclasses import asdict
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from ..events.event import SafetyEvent
from ..events.summary import aggregate_events


def deliver_webhook(
    events: list[SafetyEvent],
    url: str,
    *,
    timeout_seconds: float = 5.0,
) -> int:
    """POST a privacy-conscious event batch to an operator-configured endpoint."""

    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Webhook URL must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password:
        raise ValueError("Webhook credentials must be supplied by the integration layer, not in the URL")
    payload = {
        "schema_version": "1.0",
        "summary": aggregate_events(events),
        "events": [
            {
                **asdict(event),
                # Local evidence paths are meaningless to a remote receiver and
                # can reveal deployment layout, so never transmit them.
                "snapshot_path": None,
            }
            for event in events
        ],
    }
    request = Request(
        url,
        data=json.dumps(payload, default=str).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "utility-safety-ai/1.1"},
        method="POST",
    )
    with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310 - operator-provided integration
        return int(response.status)
