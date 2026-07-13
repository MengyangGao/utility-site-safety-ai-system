# Utility Site Safety AI

**Auditable computer vision for utility and construction worksite safety — not another helmet-detection toy.**

Utility Site Safety AI v1.1 turns image, video, webcam, and RTSP inputs into localized hazard findings, privacy-protected evidence, immutable run artifacts, and reviewable CSV/JSONL reports. It is designed as a serious engineering portfolio and proof-of-concept for substations, electrical maintenance, renewable-energy sites, EV infrastructure, and civil works.

> [!IMPORTANT]
> This is an educational and engineering demonstration prototype. It is not certified for safety-critical deployment, does not guarantee worker safety, and does not replace a safety officer. Human review is required; false positives and false negatives are expected.

![PPE-enabled video inference with normalized work zones, temporary tracking, privacy blur, and active event summary](docs/assets/annotated-video-frame-2.jpg)

*Real Pexels construction video, local PPE checkpoint, pixelated face redaction enabled. The v1.1 acceptance run processed 120 frames, wrote 468 detections, emitted two spatially distinct zone events, and saved two evidence snapshots.*

| Clean-clone person + zone mode | PPE-enabled image mode |
|---|---|
| ![General YOLO person and zone result](docs/assets/annotated-zone.jpg) | ![PPE checkpoint result](docs/assets/annotated-ppe-zone.jpg) |

Release-media provenance, run IDs, model hashes, and license boundaries are recorded in [`docs/assets/`](docs/assets/README.md).

## Why this project is different

- **Honest model capability:** a clean clone runs person + restricted-zone monitoring with COCO-pretrained YOLO11n. PPE claims activate only when the selected model actually exposes PPE classes.
- **Person-level decisions:** positive and negative PPE boxes must associate to a detected person, contradictory evidence is resolved deterministically, and unobserved PPE remains `unknown` rather than being called a violation.
- **Operational events, not frame spam:** active findings are separated from cooldown-filtered events; spatial continuity prevents short tracker-ID switches from bypassing cooldown, and findings expose opened/ongoing/resolved lifecycle state.
- **Audit-first outputs:** each run has input/model/output hashes, runtime and Git provenance, logs, summaries, detections, compliance state, media artifacts, and a collision-resistant run ID.
- **Practical interfaces:** CLI, Streamlit, image/video pipelines, webcam/RTSP ingestion, normalized zone editing, report downloads, model training, validation, benchmarking, and export.

## Two honest operating modes

| Mode | Model | What works | What does not become true automatically |
|---|---|---|---|
| Clean clone | `models/yolo11n.pt` or downloadable `yolo11n.pt` | Person detection, tracking, normalized/pixel zones, intrusion events, privacy blur, audit artifacts | Helmet, vest, gloves, boots, goggles, or `no_*` detection |
| PPE-enabled | A compatible custom checkpoint such as `models/ppe_yolo11n.pt` | The clean-clone capabilities plus the PPE classes actually present in that checkpoint | Missing PPE is never inferred merely because a positive box was absent |

The default model discovery order is:

1. `models/ppe_yolo11n.pt`
2. `models/ppe_yolo11s.pt`
3. `models/yolo11n.pt`
4. downloadable Ultralytics model name `yolo11n.pt`

The Web UI also inspects and displays the selected model's class list, hash when available, and effective capability.

## System architecture

```text
Image / Video / Webcam / RTSP
              │
              ▼
      YOLO detector adapter
   class-aware confidence floors
              │
              ▼
  tracker (Ultralytics or IoU fallback)
              │
        ┌─────┴──────────┐
        ▼                ▼
person ↔ PPE          zone geometry
 association      pixel or normalized
        └─────┬──────────┘
              ▼
       safety rule engine
  active findings + new events
              │
        privacy processing
              │
      ┌───────┼───────────┐
      ▼       ▼           ▼
 annotations  evidence    audit logs
 image/video  snapshots   CSV/JSONL/manifest
```

See [`docs/architecture.md`](docs/architecture.md) for lifecycle, trust boundaries, and extension points.

## Features

- Image, video, webcam, device-path, and RTSP inference
- Person and optional PPE detection through Ultralytics YOLO
- Person-PPE association with positive/negative conflict resolution
- Restricted zones in pixel or resolution-independent normalized coordinates
- Per-zone risk, dwell time, and required-PPE policy fields
- Deterministic `low` / `medium` / `high` / `critical` escalation
- Cooldown-based event de-duplication without hiding active findings
- Spatial event continuity across short tracker-ID changes
- Temporary per-stream track IDs; no identity recognition
- Explicit/local face localization with Gaussian, pixelated, or solid redaction and a compact head fallback
- Event, detection, compliance, and aggregate logs in JSONL and CSV
- Annotated image/video and event snapshots
- Run manifests with artifact size and SHA-256 inventory
- Failed-run manifests and atomic `latest.json` publication for successful runs
- English, Simplified Chinese, and Traditional Chinese annotations/UI
- Streamlit control room with complete/preview video modes, progress, one-click sample, zone editor, trusted-model profiles, incident review queue, compliance timeline, and complete ZIP downloads
- Automatic Web artifact retention (24 hours / 20 sessions) and trusted-model-only hosted mode
- Optional privacy-conscious JSON webhook delivery and `doctor` environment diagnostics
- Custom PPE training, validation artifacts, device benchmark, and model export
- Machine-readable model registry and aggregate/per-class promotion gate

## Quick start on a new computer

Python 3.11 is the reference environment. Python 3.10–3.12 are supported by the package metadata; Python 3.13 is intentionally excluded from the current dependency range.

```bash
conda env create -f environment.yml
conda activate utility-safety-ai
pip install -e ".[dev]"
```

For the exact direct versions used by the 2026-07-13 Python 3.11 release gate:

```bash
pip install -e ".[dev]" -c constraints-py311.txt
```

The constraints file is a known-good reference, not a universal cross-platform lock; PyTorch and
accelerator wheels still depend on the target operating system and hardware.

Fetch and pin the small general model locally. The command records its SHA-256 and installed Ultralytics version beside the checkpoint:

```bash
utility-safety-ai fetch-model --model yolo11n.pt --output models
utility-safety-ai doctor --model models/yolo11n.pt
```

The committed registry and promotion gate keep public capability claims machine-checkable. The
current local PPE checkpoint intentionally fails the sample field gate because recall is too low and
`no_vest` is absent:

```bash
utility-safety-ai model-gate \
  --metrics docs/model-evaluation/ppe_yolo11n-v1/metrics.json \
  --required-class no_helmet --required-class no_vest --required-class no_boots
```

Run the clean-clone person + zone demo on a provenance-tracked CC0 image:

```bash
utility-safety-ai infer-image \
  --source examples/sample_images/construction_zone_01.jpg \
  --model models/yolo11n.pt \
  --zones examples/zones_construction_zone_01.yaml \
  --output outputs/quickstart \
  --blur-faces
```

The command prints the exact run directory. You can also inspect the atomically updated pointer:

```bash
cat outputs/quickstart/latest.json
make show-latest OUTPUT=outputs/quickstart
```

## Run-scoped outputs

Inference never silently clears another run. By default it writes a new directory; reusing a supplied `--run-id` fails unless `--overwrite` is explicit. Explicit replacement is transactional: the old completed run remains published until the new run fully succeeds, and failed replacement evidence is isolated under `failed-runs/`.

```text
outputs/quickstart/
├── latest.json
└── runs/
    └── <run-id>/
        ├── manifest.json
        ├── images/<source>_annotated.jpg
        ├── videos/<source>_annotated.mp4
        ├── snapshots/<event-id>.jpg
        └── events/
            ├── events.jsonl
            ├── events.csv
            ├── detections.jsonl
            ├── detections.csv
            ├── compliance.jsonl
            ├── compliance.csv
            ├── summary.json
            └── summary.csv
```

Successful runs update `latest.json`; a failed run records its error but does not replace the last successful pointer. Source credentials are redacted from audit text, and privacy processing occurs before annotated media and snapshots are saved.

## CLI recipes

### Image

```bash
utility-safety-ai infer-image \
  --source path/to/image.jpg \
  --zones examples/zones_solar_inspection_pexels_4254172.yaml \
  --output outputs/image \
  --blur-faces
```

### Video

```bash
utility-safety-ai infer-video \
  --source examples/sample_videos/construction_rebar_pexels_10294768.mp4 \
  --model models/yolo11n.pt \
  --zones examples/zones_construction_rebar_pexels_10294768.yaml \
  --output outputs/video \
  --blur-faces \
  --privacy-mode pixelate
```

Video processing is complete by default. Add `--max-frames` only when an explicitly bounded preview
is intended; the manifest records the selected limit.

### Webcam or RTSP

```bash
# Webcam index 0
utility-safety-ai infer-camera \
  --source 0 \
  --output outputs/camera \
  --duration 10 \
  --blur-faces

# Credentials are redacted from manifests and text reports.
utility-safety-ai infer-camera \
  --source 'rtsp://user:password@camera.example/live?token=secret' \
  --output outputs/rtsp \
  --max-frames 300 \
  --blur-faces
```

OpenCV preview is opt-in with `--display` and requires a GUI-enabled OpenCV build. Saved video remains the normal review path for headless machines.

Send emitted events to an operator-owned integration when required:

```bash
utility-safety-ai infer-camera --source 0 --blur-faces \
  --webhook-url https://alerts.example.test/safety
```

### Reports and model export

```bash
utility-safety-ai export-report \
  --events outputs/quickstart/runs/<run-id>/events/events.jsonl \
  --format csv \
  --output outputs/quickstart-report.csv

utility-safety-ai export-model \
  --model models/ppe_yolo11n.pt \
  --format onnx \
  --output outputs/export
```

Run `utility-safety-ai --help` or `utility-safety-ai <command> --help` for all thresholds, devices, cooldown, run ID, and overwrite options.

## Web demo

```bash
streamlit run app.py
```

![Streamlit workbench with portable model path and privacy enabled by default](docs/assets/web-dashboard.jpg)

The Web UI provides:

- Image/video upload and browser camera snapshots
- Bounded RTSP/device processing and a live preview mode
- Per-session private output roots, 24-hour/20-session disk retention, and the latest 20 in-session runs
- Verified model profiles, optional trusted custom path, device, confidence, IoU, cooldown, and privacy controls
- Privacy redaction enabled by default with Gaussian, pixelated, and solid modes
- Visual-table and YAML editors for normalized zones
- Resolution-aware zone preview and downloadable zone YAML
- Annotated results, filtered risk/event tables, detection charts, PPE compliance, and evidence snapshots
- Model classes, capabilities, device, and checkpoint SHA-256 when available
- Incident acknowledgement/investigation/resolution/false-positive review states with assignee and notes
- Latest-state and full-timeline compliance views
- CSV/JSONL, separate core/Web manifests, review state, annotated media, and a complete ZIP download

RTSP fields are masked in the UI. The application also sanitizes credentials in persisted text artifacts, but operators must still protect local files, shell history, footage, and network access.

## Zone configuration

Pixel coordinates remain supported. Normalized coordinates are recommended because they scale across resolutions:

```yaml
zones:
  - id: live_switchgear
    name: Live Switchgear Boundary
    risk_level: high
    coordinate_space: normalized
    dwell_seconds: 1.5
    required_ppe: [helmet, vest, gloves]
    polygon:
      - [0.55, 0.45]
      - [0.95, 0.45]
      - [0.95, 0.95]
      - [0.55, 0.95]
```

Zone files are schema-validated. Explicitly supplied missing files, unsupported fields, invalid risk levels, duplicate IDs, degenerate polygons, and out-of-range normalized points fail fast.

## Custom PPE model

Training is optional and is not required for person + zone inference.

```bash
utility-safety-ai train \
  --data construction-ppe.yaml \
  --model yolo11n.pt \
  --epochs 30 \
  --imgsz 640 \
  --batch 16 \
  --project runs/train_ppe \
  --name ppe_yolo11n

mkdir -p models
cp runs/train_ppe/ppe_yolo11n/weights/best.pt models/ppe_yolo11n.pt
```

Then validate and benchmark the exact checkpoint:

```bash
python scripts/validate_ppe_model.py \
  --model models/ppe_yolo11n.pt \
  --data construction-ppe.yaml \
  --output outputs/validation/ppe_yolo11n

python scripts/benchmark.py \
  --model models/ppe_yolo11n.pt \
  --output outputs/benchmark/ppe_yolo11n
```

The exact locally promoted checkpoint (SHA-256 `b05d39db…cefff`) reached precision 0.6903,
recall 0.5515, mAP50 0.5786, and mAP50-95 0.2860 on the 143-image Construction-PPE validation
split. Its explicit negative classes are much weaker—`no_boots` recall was zero on four validation
instances—so this remains an engineering demonstration checkpoint, not field-ready safety evidence.
The weight is not distributed in Git and non-deterministic retraining will produce a different hash.
See the retained [evaluation artifacts](docs/model-evaluation/ppe_yolo11n-v1/README.md),
[`docs/model-card.md`](docs/model-card.md), and [`docs/dataset-card.md`](docs/dataset-card.md).

## Verification

Offline tests do not download a model or dataset:

```bash
pytest -q
ruff check .
mypy src/utility_safety_ai
python -m py_compile app.py
```

For a release-quality check:

```bash
pytest -q --cov=utility_safety_ai --cov-report=term-missing --cov-fail-under=70
python -m build
```

The final machine-specific results belong in [`REPORT.md`](REPORT.md), not in evergreen instructions.

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — processing and audit architecture
- [`docs/model-card.md`](docs/model-card.md) — intended model use and evaluation contract
- [`docs/dataset-card.md`](docs/dataset-card.md) — Construction-PPE and example-asset lineage
- [`docs/demo-script.md`](docs/demo-script.md) — repeatable Bilibili/portfolio demo flow
- [`docs/field-pilot-plan.md`](docs/field-pilot-plan.md) — site-specific data, metrics, gates, and staged validation
- [`docs/deployment.md`](docs/deployment.md) — retention, trusted hosted mode, webhooks, and multi-camera reference architecture
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — contribution workflow and quality bar
- [`SECURITY.md`](SECURITY.md) — responsible vulnerability reporting
- [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) — dependency, model, dataset, and media boundaries
- [`examples/assets.yaml`](examples/assets.yaml) — machine-readable example provenance and SHA-256 values

## Known limitations

- The default COCO model is not a PPE model.
- PPE and explicit `no_*` performance depends on the checkpoint, label balance, camera placement, distance, weather, lighting, occlusion, and site domain.
- `unknown` means unobserved; it is neither compliant nor a confirmed violation.
- Bounding-box association can mis-assign PPE when people overlap.
- The fallback tracker does not provide identity or long-occlusion re-identification.
- Track fragmentation can increase temporary-track counts. Spatial cooldown continuity suppresses overlapping ID switches, but long occlusions or materially shifted boxes may still represent a new finding.
- Zone checks use the person's bottom-center point and do not replace camera calibration or 3D site geometry.
- Local face localization and fallback redaction are privacy aids, not a complete anonymization guarantee.
- RTSP reconnect behavior is bounded; this is not a multi-camera operations platform.
- Model exports require format-specific runtimes and independent accuracy validation.
- The system is not certified under any industrial safety, cybersecurity, or privacy standard.

## Safety, privacy, and ethics

- Do not use this project for face recognition, identity inference, employee scoring, or automated discipline.
- Use temporary track IDs only within the current media stream.
- Enable privacy blur for real people and apply appropriate access control and retention rules.
- Treat every alert as a review candidate, never as proof of misconduct.
- Validate with site-specific data before any field trial and define a safe fallback when the model is unavailable.

## Resume-ready bullets

- Built an audit-oriented computer-vision safety prototype for utility and construction worksites, combining person/PPE detection, normalized hazard zones, temporary tracking, risk-ranked event aggregation, privacy-protected evidence, and run-scoped CSV/JSONL reporting.
- Designed a modular safety rule engine that resolves contradictory PPE evidence per worker, separates active findings from cooldown-filtered events, and escalates combined zone/PPE hazards deterministically.
- Delivered CPU-first CLI and Streamlit workflows with immutable run manifests, artifact hashes, RTSP credential redaction, custom YOLO training/validation/export support, and offline end-to-end tests.

## Licensing and asset provenance

The repository's original project code is offered under the root [`LICENSE`](LICENSE) (MIT). That license does **not** relicense third-party dependencies, Ultralytics software, YOLO weights, datasets, or example media.

Ultralytics currently distributes its software and YOLO models under AGPL-3.0 and commercial Enterprise terms. The Construction-PPE dataset is documented as AGPL-3.0. Anyone using this stack—especially in a proprietary, internal-company, embedded, SaaS, or commercial context—must review the applicable upstream terms and obtain an appropriate license where required. This repository does not provide legal advice or an enterprise deployment grant.

See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md), [`docs/dataset-card.md`](docs/dataset-card.md), and [`examples/assets.yaml`](examples/assets.yaml) before redistributing models, datasets, or media.
