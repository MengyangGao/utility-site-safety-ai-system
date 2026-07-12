# Security Policy

## Supported versions

Security fixes are applied to the latest `1.x` release line and the default branch. Older prototypes are not supported.

## Reporting a vulnerability

Please use GitHub's private **Report a vulnerability** / Security Advisory flow for this repository. If that flow is unavailable, contact the repository owner privately through the contact method on their GitHub profile and state that the message concerns a security vulnerability.

Do not open a public issue for:

- exposed RTSP credentials, tokens, or private camera endpoints;
- path traversal or arbitrary file access;
- unsafe model/dataset deserialization;
- command injection;
- Web session isolation failures;
- privacy blur bypasses involving real people;
- downloadable reports containing secrets or unexpected personal data;
- dependency vulnerabilities with a practical exploit path in this application.

Include a minimal reproduction, affected commit/version, impact, sanitized logs, and a suggested mitigation if available. Do not send real worksite footage, production credentials, or personal data.

The maintainer will acknowledge reports on a best-effort basis, validate impact, coordinate a fix and disclosure plan, and credit reporters who request attribution. This is a volunteer open-source project and does not promise an enterprise SLA.

## Security boundaries

This prototype does not provide:

- authentication, authorization, tenancy, or role-based access control;
- encryption at rest or a managed retention policy;
- hardened container or host configuration;
- signed models, sandboxed model loading, or supply-chain attestation;
- network segmentation, camera credential management, or secrets storage;
- high availability, disaster recovery, or certified incident response;
- safety certification or guaranteed fail-safe operation.

The Streamlit UI is appropriate for local/controlled demonstration. Do not expose it directly to an untrusted network without an authenticated reverse proxy, TLS, access controls, isolation, logging, patching, and a deployment-specific threat review.

## Operator guidance

- Use a dedicated, least-privilege camera account.
- Avoid embedding secrets in shell history; prefer a protected secret-management layer for real deployments.
- Restrict filesystem permissions on `outputs/`, because snapshots and videos may contain people.
- Keep privacy blur enabled, but assume it can miss faces.
- Review saved media before sharing it.
- Pin and verify model checksums; do not load untrusted pickle-based `.pt` files.
- Patch Python, PyTorch, Ultralytics, Streamlit, OpenCV, and transitive dependencies regularly.
- Run dependency and container scans in any deployment pipeline.
- Separate this decision-support prototype from safety interlocks or automated disciplinary processes.

## Privacy incidents

If a report or output exposes personal data:

1. Stop sharing or serving the affected artifact.
2. Revoke exposed credentials and tokens.
3. Preserve only the minimum evidence required for investigation.
4. Delete or quarantine affected outputs according to applicable policy and law.
5. Review the full input/output path, not only the blur function.
6. Notify responsible organizational privacy/security contacts where required.

This document is operational guidance, not legal advice.
