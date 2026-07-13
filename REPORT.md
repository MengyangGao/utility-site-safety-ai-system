# Utility Site Safety AI — v1.1 Release Readiness Report

**Report date:** 2026-07-13

**Target environment:** Miniconda, Python 3.11

**Release status:** v1.1 portfolio release verified on the 2026-07-13 target machine; ready for public portfolio publication, but not certified for field deployment

## Executive assessment

Utility Site Safety AI has moved beyond a single-class vision demo into an audit-oriented worksite safety prototype. It now has image, video, webcam, device, and RTSP ingestion; model-aware person/PPE behavior; resolution-independent restricted zones; temporary tracking; deterministic rules; privacy-protected evidence; run-scoped manifests; CLI and Web workflows; and offline tests.

The engineering direction is suitable for a strong open-source portfolio, technical demo, and controlled proof-of-concept. It is not production-certified and must not be represented as a substitute for safety professionals, camera calibration, cybersecurity controls, privacy governance, or site-specific model validation.

## Delivered scope

### Detection and model handling

- Ultralytics YOLO adapter for CPU, Apple MPS, or CUDA when available
- A global confidence floor enforced for every class, followed by stricter class-aware floors
- Local model discovery plus explicit model selection
- Reproducible `fetch-model` command with checkpoint SHA-256 metadata
- Clean-clone COCO person detection and restricted-zone operation
- Optional custom PPE checkpoints and training wrapper
- Model validation, benchmarking, and deployment-format export helpers

### Safety reasoning

- Person-PPE association using geometric containment/overlap
- Per-PPE positive/negative conflict resolution independent of detector order
- One resolved violation per distinct PPE type
- `unknown` state for unobserved PPE instead of fabricated violations
- Pixel and normalized polygons with strict validation
- Per-zone risk, dwell time, and required-PPE fields
- Bottom-center intrusion geometry
- Deterministic escalation across `low`, `medium`, `high`, and `critical`
- Separation of current active findings from cooldown-filtered new events

### Tracking and live inputs

- Ultralytics tracking when available and IoU fallback tracking
- Per-run tracker reset and empty-frame aging
- Temporary stream-local IDs only; no identity recognition
- Video, webcam, device-path, and RTSP processing
- Monotonic timing and bounded reconnect behavior for live sources
- Credential redaction in persisted source identifiers

### Audit outputs

- Immutable-by-default `outputs/<root>/runs/<run-id>/` layout
- Atomic `latest.json` pointer updated only by completed runs
- `manifest.json` with source/config/model context and artifact inventory
- SHA-256 and size for each completed artifact
- Failed-run manifest without replacing the last successful pointer
- Event, detection, compliance, and summary logs in CSV and JSONL
- Annotated images/videos and event evidence snapshots
- Explicit `--overwrite` requirement for run ID reuse

### Privacy and visualization

- Privacy processing before saved media and snapshots
- Exact face-region blur when face detections are associated
- Conservative upper-person fallback only when no face is associated
- No face recognition or persistent identity store
- Resolution-correct normalized zone overlays
- Bilingual/CJK-capable annotation rendering with batched frame conversion

### Web experience

- Image/video upload, browser camera snapshot, RTSP/device processing, and live preview
- Per-session output directory and bounded run history
- Privacy blur enabled by default
- Normalized zone table/YAML editor and visual preview
- Model class/capability/hash display
- Annotated media, event filters, detection charts, compliance table, evidence gallery
- Individual CSV/JSONL/manifest downloads and report ZIP
- English, Simplified Chinese, and Traditional Chinese UI support

## Clean environment setup

```bash
conda env create -f environment.yml
conda activate utility-safety-ai
pip install -e ".[dev]"
```

Optional deterministic general-model fetch:

```bash
utility-safety-ai fetch-model --model yolo11n.pt --output models
```

## Final verification record

These results were recorded on the rebuilt target environment: Python 3.11.15, PyTorch 2.13.0,
Ultralytics 8.4.92, OpenCV 4.11, and Streamlit 1.59.1 on Apple Silicon macOS.

| Gate | Command | Final result |
|---|---|---|
| Dependency consistency | `python -m pip check` | Passed — `No broken requirements found` |
| Offline tests + package coverage | `pytest -q --cov=utility_safety_ai --cov-report=term-missing --cov-fail-under=70` | Passed — 136 tests; 80.56% package coverage |
| Lint | `ruff check .` | Passed — all checks |
| Types | `mypy src/utility_safety_ai` | Passed — 42 source files |
| Web syntax/import | `python -m py_compile app.py` | Passed |
| Package build | `python -m build` | Passed — sdist and wheel created |
| Wheel install smoke | isolated venv + `pip install --no-deps dist/*.whl` + CLI/doctor | Passed — wheel version 1.1.0; CLI and diagnostics exited 0 |
| Streamlit | headless startup and Streamlit AppTest | Passed — modern control surface and required controls rendered without script exceptions; local browser navigation was blocked by the host safety policy |

The build artifacts are `dist/utility_safety_ai-1.1.0.tar.gz` and
`dist/utility_safety_ai-1.1.0-py3-none-any.whl`.

## Recorded demo acceptance

The clean-clone person + zone path was recorded with:

```bash
utility-safety-ai infer-image \
  --source examples/sample_images/construction_zone_01.jpg \
  --model models/yolo11n.pt \
  --zones examples/zones_construction_zone_01.yaml \
  --output outputs/v1.1-release \
  --run-id clean-image \
  --overwrite \
  --blur-faces
```

It completed with three detections, one zone event, one privacy-protected evidence snapshot, an
annotated image, CSV/JSONL logs, summaries, and a completed manifest under
`outputs/v1-release/runs/clean-image/`. The default COCO checkpoint validates person + zone behavior
only; it is not evidence of PPE capability.

The real-video PPE path was recorded with:

```bash
utility-safety-ai infer-video \
  --source examples/sample_videos/construction_rebar_pexels_10294768.mp4 \
  --model models/ppe_yolo11n.pt \
  --zones examples/zones_construction_rebar_pexels_10294768.yaml \
  --output outputs/v1.1-release \
  --run-id ppe-video \
  --overwrite \
  --max-frames 120 \
  --blur-faces \
  --privacy-mode pixelate
```

It completed 120 frames with 468 detections, two spatially distinct zone events, two evidence
snapshots, all audit logs, and a 1,920 x 1,080 annotated MP4 under
`outputs/v1.1-release/runs/ppe-video/`. The previous ID 3→8→9→14 fragmentation no longer emitted
three duplicate alerts inside the 10-second cooldown. A separate PPE image run under
`outputs/v1.1-release/runs/ppe-image/` completed with eight detections, one zone event, and associated
positive helmet, vest, gloves, boots, and goggles evidence.

## PPE training and evaluation status

The promoted local PPE checkpoint completed 30 epochs on Apple MPS and was independently evaluated
on CPU against the 143-image, 1,172-instance Construction-PPE validation split.

| Evidence | Result |
|---|---:|
| Checkpoint SHA-256 | `b05d39dba9d9a5a19855b9cc7dc4c613e979a64a280929e593522685c19cefff` |
| Precision | 0.6903 |
| Recall | 0.5515 |
| mAP50 | 0.5786 |
| mAP50-95 | 0.2860 |
| Detector-only CPU benchmark | 28.40 FPS / 35.21 ms |
| Detector-only Apple MPS benchmark | 117.95 FPS / 8.48 ms |

Benchmark timings use deterministic 640 x 640 synthetic input, five warm-ups, and 30 timed
`detector.predict` calls; they exclude decode, tracking, rules, privacy, annotation, logging, and
encoding. Explicit negative classes remain the principal weakness: `no_boots` recall was zero on
only four validation instances, and the dataset has no `no_vest` class. The complete metrics,
per-class results, curves, confusion matrices, environment, training configuration, and prediction
sample are retained under `docs/model-evaluation/ppe_yolo11n-v1/`. The weight is deliberately local
and excluded from Git; these claims apply only to the exact checkpoint hash.

The v1.1 `model-gate` command was run with required `no_helmet`, `no_vest`, and `no_boots` classes.
It correctly failed promotion because recall 0.5515 and mAP50 0.5786 missed the sample thresholds,
`no_helmet` recall was 0.3947, `no_boots` recall was zero, and `no_vest` was absent. The application is
release-ready as a portfolio prototype; the checkpoint is explicitly not field-ready.

## Acceptance matrix

| Requirement | Implementation status | Final evidence status |
|---|---|---|
| Fresh Python 3.11 Conda environment | Verified | Rebuilt as `utility-safety-ai`; Python 3.11.15 |
| Editable package and CLI | Verified | Editable install, sdist/wheel build, isolated wheel smoke passed |
| Image inference | Verified | `outputs/v1.1-release/runs/ppe-image/` plus retained clean-clone evidence |
| Video inference | Verified | `ppe-video/`: 120 frames, 468 detections, two events/snapshots |
| Camera/RTSP inference | Implemented | Hardware/source dependent |
| Person + zone clean-clone path | Verified | Real image run with general model and high-risk zone event |
| Optional PPE checkpoint path | Verified locally | Trained, hashed, independently validated; weight not distributed in Git |
| JSONL/CSV events and detections | Verified | Retained real runs plus offline E2E tests |
| Compliance and summaries | Verified | Retained real runs plus offline E2E tests |
| Annotated media and snapshots | Verified | Image/video artifacts visually reviewed |
| Privacy blur before persistence | Verified | Enabled in all retained release runs and covered by tests |
| Streamlit Web app | Verified with stated limitation | AppTest, headless startup and static visual QA passed; host policy blocked local-browser navigation |
| Offline tests | Verified | 136 passed; 80.56% package coverage |
| Documentation and release policies | Verified | Local Markdown links and release-asset hashes checked |

## Known limitations and residual risk

- Default YOLO11n is COCO-pretrained and does not detect PPE classes.
- A custom PPE model exposes only the classes and quality supported by its training data.
- Explicit negative PPE classes are visually ambiguous and often data-limited.
- Person-PPE association is box-based and can fail under crowding or occlusion.
- Tracking is temporary and can change IDs after long occlusion or scene cuts.
- Zone intrusion is image-plane geometry, not calibrated 3D distance.
- Local face localization and configurable redaction are not a legal anonymization guarantee and may miss small/profile/occluded faces.
- RTSP credentials can still leak through shell history, process inspection, external logs, or user screenshots even though application text artifacts are redacted.
- Export success does not prove exported-model accuracy or performance.
- A JSON webhook, bounded local retention and incident-review workflow are included. Role-based access control, durable retry queues, fleet management, HA, and safety certification are not.
- Example assets have mixed license terms with per-file hashes and verified source pages. Review `examples/assets.yaml` before redistribution.
- The root MIT license covers original project code only; third-party software, weights, datasets, and media retain their own terms.

## Recommended next improvements

1. Collect balanced, site-specific hard-negative PPE data and promote a stronger v2 checkpoint
   through the same immutable evaluation gate.
2. Re-verify example-media URLs, hashes, and license terms at each release.
3. Publish a redistributable PPE checkpoint through the committed model registry when upstream terms permit it; the current evaluated checkpoint remains local-only.
4. Add hardware-specific exported-model regression tests for ONNX/OpenVINO/TensorRT/CoreML as applicable.
5. Add authentication, authorization, encryption, durable alert retries and an external evidence store before any multi-user pilot.
6. Conduct privacy, cybersecurity, licensing, and operational safety reviews with qualified stakeholders.
7. Validate night, rain, glare, distance, crowding, occlusion, and camera-shift failure modes.

## Publication decision

The source repository is ready for public portfolio publication as a v1.1 engineering prototype.
A real worksite or commercial deployment remains a separate engineering, legal, privacy,
cybersecurity, licensing, and safety-certification program.
