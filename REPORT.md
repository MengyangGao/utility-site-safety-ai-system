# Utility Site Safety AI — v2.0 Verification Report

**Report date:** 2026-07-22

**Reference environment:** Miniconda `utility-safety-ai`, Python 3.11.15, Apple Silicon macOS
**Status:** v2.0 source, package, real-media workflows, and operator console verified for public portfolio use; not certified for field deployment

## Executive result

The project has been refactored around two outcomes: higher monitoring quality and a substantially
more polished operator experience. The v2 implementation replaces geometry-only PPE assignment,
single-frame video alerts, an iteration-order fallback tracker, and a 1,522-line Streamlit script
with body-aware association, temporal confirmation, motion-aware global tracking, explicit run-health
diagnostics, and a modular commercial-style evidence console.

The result remains an engineering prototype. The software gates pass, but the local PPE checkpoint
does not pass the documented field-promotion model gate. Human review, site-specific data, camera
calibration, privacy governance, cybersecurity controls, and operational validation remain mandatory.

## Monitoring-quality refactor

### Body-aware PPE association

- Scores PPE/person links with item containment, expected vertical body region, horizontal alignment,
  centre containment, and overlap.
- Rejects ambiguous links when the two best people score too closely instead of silently guessing.
- Keeps deterministic positive/negative conflict resolution and `unknown` for absent evidence.
- Records association coverage as an operational diagnostic, not as a model-accuracy claim.

### Temporal alert confirmation

- Separates active, provisional, confirmed, newly emitted, and resolved findings.
- Adds `confirmation_count`, `confirmation_required`, and `confirmation_status` to event metadata.
- Keeps still-image findings immediate; only temporal sources accumulate consecutive evidence.
- Provides three explicit presets:

| Profile | PPE confirmation | Zone confirmation | Intended trade-off |
|---|---:|---:|---|
| `balanced` | 3 frames | 2 frames | Everyday review stability |
| `high_precision` | 5 frames | 3 frames | Fewer interruptions, stronger evidence |
| `high_sensitivity` | 2 frames | 1 frame | Earlier investigation, higher recall bias |

### Motion-aware fallback tracking

- Uses bounded linear prediction, IoU, and normalized centre distance.
- Ranks all track/detection candidates globally to avoid iteration-order detection stealing.
- Preserves upstream Ultralytics IDs, ages short misses, and remains temporary/run-local only.
- Does not perform identity recognition, re-identification, or cross-camera tracking.

### Run health evidence

Every successful image, video, and camera run now includes `quality.json` and `quality.csv` with:

- tracked-person observation rate;
- PPE assignment rate and unassociated PPE count;
- provisional and confirmed finding observations;
- temporal filter rate;
- inference and effective processing throughput;
- explicit interpretation text preventing these indicators from being mistaken for accuracy.

## Operator-console refactor

The root `app.py` is now a five-line entry point. Product configuration, session isolation, visual
theme, inference services, orchestration, result rendering, and review persistence live in focused
`utility_safety_ai.web` modules.

The interface now provides:

- a four-stage input → policy → analysis → review mental model;
- an industrial dark visual system with amber safety accents, responsive cards, and consistent spacing;
- balanced/high-precision/high-sensitivity monitoring profiles;
- verified model profiles plus an explicitly warned advanced custom path;
- CPU as the dependable default, with Auto, Apple MPS, and CUDA as explicit alternatives;
- privacy blur enabled by default and three redaction styles;
- image, complete/preview video, browser snapshot, camera, and RTSP workflows;
- normalized zone policy editing and visual preview;
- evidence media, run integrity, checkpoint hash, quality indicators, event review, operator notes,
  complete audit ZIP download, and private session history.

The browser acceptance pass exposed a real issue: automatic Apple MPS selection could spend too long
in first-use initialization on the target machine. The final interface therefore defaults to CPU,
matching the project's documented clean-clone reliability requirement while retaining explicit
accelerator choices.

## Final verification gates

| Gate | Command | Result |
|---|---|---|
| Environment version | editable reinstall + metadata check | Passed — distribution and package both `2.0.0` |
| Dependencies | `python -m pip check` | Passed — no broken requirements |
| Tests and coverage | `pytest -q --cov=utility_safety_ai --cov-report=term-missing --cov-fail-under=70` | Passed — 146 tests, 76.78% coverage |
| Lint | `ruff check .` | Passed |
| Types | `mypy src/utility_safety_ai` | Passed — 52 source files |
| Web syntax | `python -m py_compile app.py` | Passed |
| Workflow syntax | `actionlint .github/workflows/ci.yml` | Passed |
| Distribution build | `python -m build --no-isolation` | Passed — sdist and wheel |
| Wheel smoke | isolated system-site venv, wheel install, import/version, CLI help | Passed — imported from isolated venv as `2.0.0` |
| Streamlit smoke | `Streamlit AppTest` | Passed — required controls rendered, privacy default ON |
| Browser visual/interaction | local headless server + real in-app browser | Passed — console, one-click sample, result navigation, quality, bundle, and review controls |

Built artifacts:

- `dist/utility_safety_ai-2.0.0.tar.gz`
- `dist/utility_safety_ai-2.0.0-py3-none-any.whl`

## Recorded real-media acceptance

### PPE image

```bash
utility-safety-ai infer-image \
  --source examples/sample_images/construction_site_ppe_01.jpg \
  --model models/ppe_yolo11n.pt \
  --zones examples/zones_construction_site_ppe_01.yaml \
  --output outputs/v2-acceptance \
  --run-id ppe-image \
  --device cpu \
  --profile balanced \
  --privacy-mode pixelate \
  --blur-faces
```

Result: one frame, eight detections, one confirmed zone event, one privacy-processed snapshot,
completed manifest, all CSV/JSONL logs, annotated image, and quality artifacts under
`outputs/v2-acceptance/runs/ppe-image/`. Tracking and PPE assignment coverage were both 100% for
this frame; that describes pipeline linkage only, not ground-truth accuracy.

### PPE video

```bash
utility-safety-ai infer-video \
  --source examples/sample_videos/construction_ppe_pan.mp4 \
  --model models/ppe_yolo11n.pt \
  --zones examples/zones_construction_site_ppe_01.yaml \
  --output outputs/v2-acceptance \
  --run-id ppe-video \
  --device cpu \
  --profile balanced \
  --privacy-mode pixelate \
  --blur-faces \
  --max-frames 120
```

The source contained 75 decodable frames, all of which were processed. The run wrote 438 detections,
one confirmed event/snapshot, a validated annotated MP4, and all required logs. Operational quality
evidence recorded one unique temporary track, 100% tracked-person observation coverage, 100% PPE
assignment coverage, one provisional finding observation, 74 confirmed finding observations, and
4.16 effective CPU FPS including annotation and persistence. These are run-health measurements, not
precision/recall metrics.

### Browser workflow

The final CPU-default console was loaded in a real browser. The portfolio sample completed, displayed
one confirmed event, and the workspace then rendered the annotated image, four summary metrics,
model capabilities and truncated checkpoint hash, quality indicators, complete audit-bundle download,
evidence snapshot, editable review decision, and operator note field. Desktop visual layout was also
inspected directly.

## Model evidence and capability boundary

The local checkpoint hash remains:

`b05d39dba9d9a5a19855b9cc7dc4c613e979a64a280929e593522685c19cefff`

Its retained validation evidence remains precision 0.6903, recall 0.5515, mAP50 0.5786, and
mAP50-95 0.2860 on the documented 143-image validation split. `no_boots` recall was zero on four
instances and `no_vest` is absent from the dataset. The `model-gate` correctly rejects this checkpoint
for field promotion. The software is portfolio-ready; the checkpoint is not field-ready.

## Known limitations

- COCO YOLO11n supports person and zone monitoring, not PPE.
- PPE results are limited to the exact selected checkpoint classes and dataset quality.
- Body-aware 2D association is safer in crowds but can still fail under severe overlap or occlusion.
- Consecutive-frame confirmation reduces transient alert noise but can delay true alerts by a few frames.
- Temporary tracking can still fragment after long occlusion, abrupt cuts, or large motion.
- Zone geometry is image-plane logic, not calibrated 3D distance.
- Privacy redaction is an aid, not a legal anonymisation guarantee.
- Camera/RTSP hardware, credentials, network resilience, multi-user authentication, durable alert queues,
  and external evidence storage require deployment-specific engineering.
- No software result guarantees worker safety or replaces qualified safety professionals.

## Recommended next work

1. Collect balanced site-specific hard negatives and explicit missing-PPE examples, especially
   `no_vest` and `no_boots`, then retrain and pass the unchanged promotion gate.
2. Add a labelled temporal evaluation set measuring event precision, event recall, time-to-alert, and
   track fragmentation across crowding, glare, rain, night, occlusion, and camera motion.
3. Calibrate monitoring profiles per deployment only after that labelled evaluation.
4. Add authenticated multi-user review, encrypted external evidence storage, and durable alert delivery
   before any controlled pilot.
5. Run privacy, cybersecurity, licensing, camera-placement, and operational safety reviews with
   qualified stakeholders.

## Publication decision

The v2.0 source is suitable for public portfolio publication and demonstration as an auditable
engineering prototype. A real worksite or commercial deployment remains a separate model,
engineering, privacy, cybersecurity, legal, and safety-certification programme.
