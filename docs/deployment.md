# Deployment and operations

## Safe local portfolio mode

```bash
conda activate utility-safety-ai
utility-safety-ai doctor
streamlit run app.py
```

The Web app stores each browser session beneath `outputs/web_demo/sessions/`. It automatically
removes sessions older than 24 hours and enforces a 20-session ceiling. Uploaded temporary files are
deleted after each pipeline call. Treat generated evidence as sensitive even with privacy redaction.

## Hosted demonstration mode

Set `UTILITY_SAFETY_TRUSTED_MODELS_ONLY=1` to disable arbitrary local checkpoint paths. Put the app
behind TLS and authentication, isolate it from internal networks, run it as a non-privileged user,
and set platform-level CPU, memory, upload and request-time limits. Never expose a raw Streamlit
process directly to the internet.

Optional alert delivery is available to an operator-owned endpoint:

```bash
export UTILITY_SAFETY_WEBHOOK_URL=https://alerts.example.test/safety
utility-safety-ai infer-camera --source 0 --zones examples/zones.example.yaml
```

Webhook payloads exclude local snapshot paths. Authentication, retry queues and secrets should be
provided by the deployment integration layer rather than embedded in source URLs.

## Multi-camera production reference

For a real pilot, run camera ingestion and inference as supervised workers, publish sanitized event
messages to a durable queue, and keep the dashboard stateless. Add per-camera heartbeats, reconnect
metrics, backlog/latency alerts, dead-letter handling and an authenticated evidence store. The
Streamlit application remains the portfolio/control-room prototype; it is not a multi-tenant camera
orchestration service.

## Operator workflow

The Web event tab supports `new`, `acknowledged`, `investigating`, `resolved` and `false_positive`
review states with assignee and notes. These decisions are written to `review_state.json` and included
in the complete ZIP export. They are local audit aids, not an identity or personnel-management system.
