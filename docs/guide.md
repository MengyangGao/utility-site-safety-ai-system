# Product and operations guide

## Architecture

```text
image / video / camera / RTSP
              │
              ▼
      YOLO detector adapter
              │
              ▼
   motion-aware temporary tracker
          ┌───┴────────┐
          ▼            ▼
  person ↔ PPE     zone geometry
          └───┬────────┘
              ▼
        rule lifecycle
 provisional → confirmed → resolved
              │
         privacy redaction
              │
       annotated evidence
              │
 events · compliance · quality · manifest
```

The detector adapter normalizes model output. Tracking remains local to one stream and is never an
identity claim. PPE boxes are linked to people only when their geometry is plausible and the best
candidate is not ambiguous. Zone rules use each person's bottom-center point.

For videos and streams, findings pass through profile-controlled consecutive-frame confirmation.
Cooldown suppresses duplicate log entries without hiding an active finding. Images are evaluated
immediately because they cannot accumulate temporal evidence.

## Monitoring profiles

| Profile | Best for | Trade-off |
|---|---|---|
| `balanced` | Normal demos and review | Default balance |
| `high_precision` | Reducing weak/transient alerts | More missed or delayed findings |
| `high_sensitivity` | Surfacing weak evidence early | More false positives |

These profiles tune model confidence, overlap suppression, person-PPE association, tracking memory,
and confirmation frames. They are engineering presets, not calibrated safety guarantees.

## Interfaces

Show all commands:

```bash
utility-safety-ai --help
```

Common workflows:

```bash
# Web console
utility-safety-ai web

# Image
utility-safety-ai infer-image --source image.jpg --zones zones.yaml --blur-faces

# Video
utility-safety-ai infer-video --source video.mp4 --zones zones.yaml --blur-faces

# Camera index or RTSP
utility-safety-ai infer-camera --source 0 --max-frames 300 --blur-faces

# Report conversion
utility-safety-ai export-report --events events.jsonl --format csv --output report.csv
```

The Web console provides image/video upload, browser capture, camera/RTSP input, normalized-zone
editing, model capability inspection, privacy controls, evidence review, operator notes, history,
and complete audit ZIP downloads. Privacy redaction and CPU processing are the stable defaults.

## Zone policy

Zones accept pixel coordinates or normalized values in `[0, 1]`. Normalized policies are preferred
because they remain aligned when input resolution changes.

```yaml
zones:
  - id: work_area
    name: Restricted work area
    coordinate_space: normalized
    risk_level: high
    min_dwell_seconds: 0.5
    required_ppe: [helmet, vest]
    polygon:
      - [0.15, 0.40]
      - [0.85, 0.40]
      - [0.90, 0.95]
      - [0.10, 0.95]
```

Image-plane polygons are not calibrated physical distances. Camera movement invalidates the policy
until it is reviewed and realigned.

## Run lifecycle

Every run receives an isolated directory. Reusing a run ID fails unless `--overwrite` is explicit.
Replacement is transactional: a failed attempt is written below `failed-runs/` and cannot replace
the last successful `latest.json` pointer.

Persisted source strings redact RTSP user information and common token query parameters. Privacy
processing occurs before annotated media or snapshots are written. Treat all output as sensitive;
redaction can still miss identifying details.

## Deployment boundary

The Streamlit app is intended for a local or controlled demonstration. For a hosted demo:

- set `UTILITY_SAFETY_TRUSTED_MODELS_ONLY=1`;
- place the service behind TLS, authentication, and access controls;
- run as an unprivileged user with upload, CPU, memory, and time limits;
- protect `outputs/` and define retention/deletion rules;
- load only trusted model files—PyTorch checkpoints can execute code;
- keep camera credentials in a deployment secret store, not source URLs or shell history.

Real multi-camera use needs supervised workers, reconnect handling, a durable queue, camera health,
backlog/latency monitoring, an authenticated evidence store, and incident procedures. This project
does not supply those production controls.

## Security and privacy

Report vulnerabilities through the repository's private GitHub Security Advisory flow. Do not put
credentials, identifiable worksite footage, private weights, or proprietary datasets in a public
issue. High-priority concerns include unsafe model loading, path traversal, credential leakage,
privacy-redaction bypass, session isolation, and report exposure.

This project deliberately excludes face recognition, persistent identity, automated discipline,
authentication, tenancy, and safety interlocks.
