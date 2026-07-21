# v2.1 verification record

Date: 2026-07-22
Reference environment: macOS arm64, Python 3.11, `utility-safety-ai` Conda environment

## Release scope

Version 2.1 aligns the repository with an open-source-only publication policy:

- project license changed to `AGPL-3.0-only`;
- Pexels/custom-stock and unverifiable-source media removed;
- untraceable validation mosaics removed;
- 15 retained media/evaluation artifacts registered with source, license, and SHA-256;
- provenance audit added to CLI, tests, and CI;
- root reduced to five tracked files;
- README reduced and detailed documentation consolidated;
- Streamlit entry moved into the package and exposed as `utility-safety-ai web`.

## Automated gates

| Gate | Result |
|---|---|
| `pytest -q --cov=utility_safety_ai --cov-report=term-missing --cov-fail-under=70` | 150 passed; 77.08% coverage |
| `ruff check .` | Passed |
| `mypy src/utility_safety_ai` | Passed; 54 source files |
| `python -m py_compile src/utility_safety_ai/web/app.py` | Passed |
| `actionlint` | Passed |
| `pip check` | Passed |
| `utility-safety-ai audit-provenance` | Passed; 15 artifacts; no errors |
| `python -m build --no-isolation` | sdist and wheel built |
| Installed metadata | package `2.1.0`; license expression `AGPL-3.0-only` |
| Streamlit `AppTest` | Passed; privacy default and primary controls rendered |
| Isolated wheel smoke | CLI and package-internal Web app passed outside the checkout |

## Open-source media acceptance

Both runs used the exact AGPL-3.0 Construction-PPE sample recorded in
[`legal/provenance.yaml`](legal/provenance.yaml) and the general `yolo11n.pt` detector.

### Image

```bash
utility-safety-ai infer-image \
  --source examples/sample_images/construction_site_ppe_01.jpg \
  --model models/yolo11n.pt \
  --zones examples/zones_construction_site_ppe_01.yaml \
  --output outputs/v2.1-acceptance \
  --run-id open-source-image \
  --overwrite --profile balanced --blur-faces
```

Result: completed; 1 frame, 1 person detection, 1 zone event, 100% temporary-track coverage, 12
hashed artifacts.

### Video

```bash
utility-safety-ai infer-video \
  --source examples/sample_videos/construction_ppe_pan.mp4 \
  --model models/yolo11n.pt \
  --zones examples/zones_construction_site_ppe_01.yaml \
  --output outputs/v2.1-acceptance \
  --run-id open-source-video \
  --overwrite --profile balanced --max-frames 15 --blur-faces
```

Result: completed; 15 frames, 15 person detections, 1 confirmed zone event, 1 provisional
observation filtered before confirmation, 100% temporary-track coverage, 3.61 full-pipeline CPU
FPS, and 12 hashed artifacts.

These operational indicators are not accuracy, identity, or field-readiness metrics.

## Model boundary

The clean-clone model supports person and restricted-zone monitoring, not PPE. The retained local
PPE checkpoint still fails promotion: recall `0.5515`, mAP50 `0.5786`, zero `no_boots` recall, and no
`no_vest` class. It remains local-only engineering evidence.

## Remaining release obligations

- Generate an SBOM/license report for each final binary or container, including transitive native
  libraries and codecs.
- Re-run dependency, model, dataset, and privacy review for any hosted or commercial release.
- Perform site-specific labeled evaluation before even a supervised field pilot.
- Obtain qualified legal advice when organizational deployment terms matter.
