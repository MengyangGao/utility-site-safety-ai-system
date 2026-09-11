# Integration contracts

All integrations are optional and operator configured. The default application performs local
inference and stores events. The SQLite database is `<output-root>/events.sqlite3`; the dashboard
uses its isolated session output root.

## Webhook

Set `UTILITY_SAFETY_WEBHOOK_URL`, optionally `UTILITY_SAFETY_WEBHOOK_SECRET` and
`UTILITY_SAFETY_WEBHOOK_TOKEN`, before running a CLI inference command. `--webhook-url` overrides
the URL. A worker sends newly recorded events while inference continues.

Requests contain an `events` array and count summary in schema version `1.1`. The durable worker
sends one event per request; the Python `deliver_webhook` helper also accepts explicit batches.
`source_path` and `snapshot_path` are always null, and only allowlisted metadata is included.
Evidence bytes, camera credentials and arbitrary metadata are not sent.

With a signing secret configured:

```text
X-Safety-Timestamp: <Unix seconds>
X-Safety-Signature: sha256=<hex HMAC-SHA256(secret, timestamp + "." + exact request bytes)>
Idempotency-Key: <stable SHA-256 of the ordered event identifiers>
Authorization: Bearer <optional token>
```

Verify the signature using the exact received body, enforce a timestamp window, and deduplicate
the idempotency key. Signatures alone do not prevent replay. Remote webhooks require HTTPS;
loopback HTTP is accepted for development. Redirects are rejected.

## Delivery semantics

A successful HTTP 2xx or MQTT broker acknowledgement completes delivery. Network failures,
HTTP 408/429 and server failures retry with exponential backoff, capped at 60 seconds. Permanent
HTTP failures or six unsuccessful attempts move a record into the `dead` state. Event metadata
remains in the local database. Expired worker leases can be recovered after a process restart.
This is at-least-once delivery, so consumers must deduplicate.

```bash
# Send currently due records using the endpoints in the current environment.
uv run utility-safety-ai deliver-pending --output outputs/edge

# Explicitly requeue dead-letter records after fixing configuration.
uv run utility-safety-ai deliver-pending --output outputs/edge --retry-failed
```

The retry command handles records due now; invoke it again after the backoff interval or keep
an inference producer running to service its outbox. Configuration is not persisted in the
outbox. Changing an endpoint routes pending records to the newly configured receiver.

## MQTT

Install with `uv sync --locked --extra mqtt`, then use `uv run --extra mqtt ...`.

| Variable | Default |
|---|---|
| `UTILITY_SAFETY_MQTT_HOST` | unset; MQTT disabled |
| `UTILITY_SAFETY_MQTT_PORT` | 8883 with TLS, 1883 without |
| `UTILITY_SAFETY_MQTT_TLS` | `1`; only loopback may use `0` |
| `UTILITY_SAFETY_MQTT_TOPIC` | `utility-safety/events` |
| `UTILITY_SAFETY_MQTT_USERNAME` / `UTILITY_SAFETY_MQTT_PASSWORD` | unset |

The payload is one sanitized event object. Messages use QoS 1 and `retain=false`; use `event_id`
for deduplication. Broker acknowledgement means the broker accepted the message, not that a
remote application or human reviewed it.

## REST API

```bash
export UTILITY_SAFETY_API_TOKEN='replace-with-a-random-long-token'
uv run --extra api utility-safety-ai api --output outputs/edge --port 8080
```

The service binds to `127.0.0.1`. Every data endpoint requires `Authorization: Bearer <token>`.
The token must contain at least 24 characters. API docs are at `/docs`.

| Endpoint | Behaviour |
|---|---|
| `GET /health` | Database connectivity and delivery-state counts |
| `GET /events?after=0&limit=100` | Sanitized events and `next_cursor`; optional `run_id` filter |
| `GET /events/{event_id}/evidence` | Protected local snapshot, with path containment checks |
| `POST /events/{event_id}/reviews` | Append `{status, note, reviewer}` |
| `GET /events/{event_id}/reviews` | Ordered review history |

Review statuses are `unreviewed`, `confirmed`, `false_positive` and `needs_follow_up`.
A reviewer label is supplied by the caller; the shared API token is not a per-user identity
system. Put a separately managed authentication/TLS boundary in front of this service if
remote access is needed. The application does not start that boundary automatically.
