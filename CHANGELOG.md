# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases use [Semantic Versioning](https://semver.org/).

## [Unreleased]

- No unreleased changes.

## [1.0.0] - 2026-07-13

### Added

- Image, video, webcam/device, and RTSP inference through one CLI.
- Streamlit dashboard with upload/camera/live modes, multilingual UI, normalized zone editor, model capability audit, run history, evidence gallery, and report downloads.
- Person-PPE association and per-person compliance logs.
- Normalized zone coordinates, per-zone risk, dwell time, and required-PPE fields.
- `RuleEvaluation` interface separating active findings from cooldown-filtered new events.
- Run-scoped output directories, atomic `latest.json`, success/failure manifests, and artifact SHA-256 inventory.
- Event, detection, compliance, and summary artifacts in CSV and JSONL.
- Event evidence snapshots and annotated image/video artifacts.
- RTSP source credential redaction and per-session Web output roots.
- `fetch-model`, report export, custom training, validation, benchmark, and model-export workflows.
- Auditable local PPE-checkpoint evidence with checkpoint/dataset hashes, aggregate and per-class
  validation metrics, confusion matrices, curves, prediction samples, and scoped CPU/MPS benchmarks.
- Offline synthetic end-to-end coverage for core inference pipelines.
- Open-source governance, model/dataset cards, architecture, release report, and demo guide.

### Changed

- Default capability is now described honestly as COCO person + zone monitoring unless a PPE-capable checkpoint is loaded.
- Positive/negative PPE conflicts resolve once per person/PPE type; repeated glove/boot boxes no longer create artificial multi-violation escalation.
- The global confidence is enforced for every class; stricter class-specific floors are applied afterward.
- Removed the extra post-model NMS pass that could suppress nearby workers or PPE.
- Tracking and rule state reset at every run; empty frames age fallback tracks.
- Privacy processing occurs before persisted outputs and snapshots.
- Exact associated face boxes are preferred over broad upper-person fallback blur.
- Zone configuration errors fail fast instead of silently disabling protection.
- Output reuse requires explicit `--overwrite`.

### Fixed

- PPE events no longer consume contradicted raw negative detections after resolved compliance is positive.
- Live preview state no longer stops immediately after start or shares tracker/rule state across runs.
- Repeated active hazards stay visible in annotations while cooldown suppresses only duplicate log emissions.
- Normalized zones are evaluated and drawn at the source resolution.
- Failed runs do not replace the most recent successful pointer.
- Uploaded temporary files and Web session outputs follow bounded, isolated lifecycles.
- CJK text rendering batches full-frame color conversion once per annotated frame.

### Security and privacy

- Privacy blur defaults to enabled in the Web UI.
- RTSP user-info and sensitive query values are redacted from persisted source text.
- Face recognition and persistent identity tracking remain intentionally out of scope.
- Model, dataset, dependency, and media licensing boundaries are documented separately from the root project-code license.

### Known limitations

- The default COCO checkpoint does not detect PPE.
- The retained local PPE checkpoint is an engineering-demonstration result scoped to its exact
  SHA-256 and validation artifacts; it is not field-ready safety evidence and does not transfer to a
  clean clone or newly retrained weight.
- This release is not certified for safety-critical, privacy-regulated, or production deployment.

[Unreleased]: https://github.com/MengyangGao/utility-site-safety-ai-system/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/MengyangGao/utility-site-safety-ai-system/releases/tag/v1.0.0
