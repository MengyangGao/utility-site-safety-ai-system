# Utility Site Safety AI

**Auditable visual monitoring for utility and construction worksites.**

Utility Site Safety AI turns images, videos, cameras, and RTSP streams into tracked safety findings,
privacy-protected evidence, and reviewable audit reports. It combines detection, work-zone geometry,
temporal confirmation, and human review in one CPU-compatible open-source prototype.

> [!WARNING]
> This project is not safety-certified and does not replace a safety officer. False positives and
> false negatives are expected. Human review is required.

![Privacy-protected PPE and restricted-zone result](docs/assets/annotated-ppe-zone.jpg)

## What it does

- Detects people with the clean-clone YOLO11n path; enables PPE rules only for compatible checkpoints.
- Associates PPE to individual people using body region, containment, alignment, and ambiguity gates.
- Detects entry into normalized polygon zones and confirms video findings across multiple frames.
- Preserves temporary track continuity without face recognition or persistent identity.
- Blurs privacy-sensitive regions before saving annotated media and evidence.
- Writes events, detections, compliance, quality metrics, summaries, hashes, and a run manifest.
- Provides a modern Streamlit review console plus reproducible CLI workflows.

## Capability boundary

| Setup | Supported | Not implied |
|---|---|---|
| Clean clone + `yolo11n.pt` | Person detection, tracking, zones, privacy, audit artifacts | PPE detection |
| Compatible custom PPE checkpoint | The above plus classes exposed by that checkpoint | Field readiness or complete PPE coverage |

The retained local PPE evaluation is intentionally **not promoted**: recall is `0.5515`, mAP50 is
`0.5786`, `no_boots` recall is zero, and the source dataset has no `no_vest` class. See the
[model and data record](docs/model-and-data.md).

## Quick start

Python 3.11 is the reference environment; Python 3.10–3.12 are supported.

```bash
conda env create -f config/environment.yml
conda activate utility-safety-ai
pip install -e ".[dev]"
```

Download and record the general detector:

```bash
utility-safety-ai fetch-model --model yolo11n.pt --output models
utility-safety-ai doctor --model models/yolo11n.pt
```

Launch the operator console:

```bash
utility-safety-ai web
```

Run the open-source sample:

```bash
utility-safety-ai infer-image \
  --source examples/sample_images/construction_site_ppe_01.jpg \
  --model models/yolo11n.pt \
  --zones examples/zones_construction_site_ppe_01.yaml \
  --output outputs/demo \
  --profile balanced \
  --blur-faces
```

Video uses the same policy file:

```bash
utility-safety-ai infer-video \
  --source examples/sample_videos/construction_ppe_pan.mp4 \
  --model models/yolo11n.pt \
  --zones examples/zones_construction_site_ppe_01.yaml \
  --output outputs/video \
  --blur-faces
```

## Audit output

Each successful run is isolated under `outputs/<name>/runs/<run-id>/` and contains:

```text
manifest.json                 status, configuration, model/input/output hashes
quality.json / quality.csv    tracking, association, filtering, and throughput indicators
events/                       events, detections, compliance, and summaries in JSONL/CSV
images/ or videos/            privacy-processed annotated result
snapshots/                    event evidence
```

`latest.json` changes only after a run completes successfully. Operational quality indicators are
not ground-truth accuracy metrics.

## Verification

```bash
pytest -q --cov=utility_safety_ai --cov-report=term-missing --cov-fail-under=70
ruff check .
mypy src/utility_safety_ai
utility-safety-ai audit-provenance
python -m build
```

The test suite is offline and does not download models or datasets. Current machine-specific
evidence is recorded in [docs/verification.md](docs/verification.md).

## Open-source and provenance policy

The project is licensed under **AGPL-3.0-only**, aligned with its integrated Ultralytics stack.
Committed third-party media and evaluation artifacts must use an approved open-source/public-domain
license, have an exact origin or derivation, and match a recorded SHA-256. CI enforces this with
`utility-safety-ai audit-provenance`.

- [Machine-readable provenance](docs/legal/provenance.yaml)
- [Third-party notices](docs/legal/third-party-notices.md)
- [GNU AGPL v3](LICENSE)

No Pexels or other custom stock-license media is included. Model weights and downloaded datasets
remain excluded from Git; their upstream terms still apply. This summary is not legal advice.

## Documentation

- [Product, architecture, CLI, Web, and deployment guide](docs/guide.md)
- [Models, datasets, training, evaluation, and field gates](docs/model-and-data.md)
- [Development and contribution guide](docs/development.md)
- [Release verification](docs/verification.md)
- [Changelog](docs/changelog.md)

## Responsible use

Do not use this prototype for autonomous shutdown, disciplinary decisions, covert surveillance,
face recognition, or certified safety control. Validate every camera, class, zone, privacy path,
and operating condition before any supervised pilot.
