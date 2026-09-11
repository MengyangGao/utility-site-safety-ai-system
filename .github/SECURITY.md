# Security policy

Security fixes are applied to the latest release and the default branch.

## Report a vulnerability

Use GitHub's private **Report a vulnerability** / Security Advisory flow. Please do not open a
public issue for exposed camera credentials, unsafe model loading, path traversal, command injection,
privacy-redaction bypasses, session isolation problems, or reports that expose private data.

Include the affected version, a minimal reproduction, impact, and sanitized logs. Never attach real
worksite footage, credentials, private model weights, or personal data.

## Safe operation

- Keep the Web console on a local or access-controlled network.
- Keep privacy redaction enabled and review media before sharing it.
- Protect `outputs/`, camera URLs, tokens, and report downloads.
- Load only trusted model files; PyTorch `.pt` files can contain executable code.
- Update Python, PyTorch, Ultralytics, Streamlit, and OpenCV regularly.

## Local runtime and integrations

The model loader disables Ultralytics usage events in memory and defaults local inference to
`YOLO_OFFLINE=1` before importing the runtime. It does not rewrite global Ultralytics settings.
Explicit model/dataset download commands may enable network access when the operator has not
set an offline override. Streamlit usage statistics are disabled by the CLI and repository config.

Webhooks and MQTT are operator-configured outbound integrations. Their allowlisted event payloads
exclude source URLs, snapshot paths and raw media. The REST API requires a bearer token and binds
to loopback. Review labels are caller-supplied, not independently authenticated identities.

Automatic face/head redaction is best effort and may miss people or other identifying details.
Run databases and evidence are unencrypted local files. Session cleanup runs when sessions start;
it is not a timer-driven secure-erasure guarantee. Native camera backend debug logs are outside
the Python log redaction layer; avoid enabling verbose codec logging on credential-bearing streams.
