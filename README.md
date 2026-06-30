# 🦺 Utility Site Safety AI System

An open-source computer vision prototype for safety monitoring in utility, construction, substation, electrical maintenance, and infrastructure worksite scenarios.

> ⚠️ **Disclaimer**: This project is an **educational and engineering demonstration prototype**. It is **not certified** for real safety-critical deployment. It is intended as decision-support software that requires human review.

---

## What's new in v0.4 (industrial-deliverable prototype)

- **Person-level PPE compliance reports** — geometry-based association of PPE items to each detected worker produces per-person status: `helmet yes`, `vest yes`, `gloves no`, etc.
- **Live camera / RTSP / webcam inference** — `infer-camera` CLI command and Streamlit tab for continuous input sources.
- **Comprehensive validation artifacts** — per-class precision/recall/mAP50/mAP50-95, confusion matrices, PR/F1 curves, precision/recall curves, and validation-batch prediction galleries.
- **Performance benchmarking** — `scripts/benchmark.py` measures FPS on CPU and available accelerators (MPS/CUDA).
- **Model export helper** — `scripts/export_model.py` exports trained weights to ONNX/TorchScript/OpenVINO/TensorRT formats.
- **Cleaner project workspace** — obsolete outputs, cached files, and dead configs removed; repo reorganized for maintainability.
- **Honest negative-PPE handling** — events are generated only from explicit `no_*` detections or geometry-based person-PPE association; the system never fakes a violation from a missing positive detection.

---

## Use Cases

- Substation maintenance monitoring
- Utility construction site safety
- Electrical cabinet / switchgear work observation
- Solar / wind farm construction and O&M
- EV charging facility construction
- General infrastructure worksite safety

---

## Features

- ✅ **PPE detection** — person, helmet, vest, gloves, boots, goggles
- ✅ **Explicit missing-PPE detection** — `no_helmet`, `no_goggle`, `no_gloves`, `no_boots` (dataset-dependent)
- ✅ **Person-level PPE compliance** — geometry-based association reports per worker
- ✅ **Person detection fallback** — general-person detection with `yolo11n.pt` when no PPE model is present
- ✅ **Restricted-zone intrusion detection** — configurable polygonal zones with per-zone risk levels
- ✅ **Risk-level rule engine** — deterministic escalation: low → medium → high → critical
- ✅ **Event de-duplication** — cooldown-based merging of repeated violations
- ✅ **Event logging** — every event to JSONL and CSV in `outputs/events/`
- ✅ **Detection audit logs** — every detection to JSONL and CSV
- ✅ **Compliance logs** — per-person PPE status to JSONL and CSV
- ✅ **Event aggregation** — per-run `summary.json` / `summary.csv`
- ✅ **Annotated image/video output** — bounding boxes, labels, zones, and risk summaries
- ✅ **Privacy-preserving face/person blur** — optional upper-body / face-region blurring
- ✅ **Web demo** — Streamlit app with upload, webcam snapshot, RTSP/live camera, zone editor, reports
- ✅ **CLI tools** — `infer-image`, `infer-video`, `infer-camera`, `train`, `export-report`, `export-model`
- ✅ **Validation suite** — per-class metrics, confusion matrix, PR/F1 curves, batch galleries
- ✅ **Benchmarking** — FPS on CPU/MPS/CUDA
- ✅ **Model export** — ONNX/TorchScript/OpenVINO/TensorRT
- ✅ **YOLO fine-tuning support** — wrapper for custom PPE / safety datasets

---

## Architecture

```text
Image / Video / Camera
         │
         ▼
   YOLO Detector
         │
         ▼
   Tracker (ByteTrack / BoT-SORT or IoU fallback)
         │
         ▼
   Zone Intrusion Checker
         │
         ▼
   Safety Rule Engine
         │
         ▼
   Event Logger + Annotator + Summary Reporter
         │
         ▼
   Dashboard / Reports / Annotated Outputs
```

---

## Setup

Requires [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (or Anaconda).

> **Python version**: Tested on Python 3.10–3.12. Python 3.13 is not yet supported
> because numpy 1.x does not provide wheels for it.

```bash
conda env create -f environment.yml
conda activate utility-safety-ai
pip install -e ".[dev]"
```

The default inference model is auto-discovered in this order:

1. `models/ppe_yolo11n.pt` if present — fast, CPU-friendly PPE model (default).
2. `models/ppe_yolo11s.pt` if present — slightly heavier, marginally stronger PPE model.
3. `yolo11n.pt` otherwise — Ultralytics downloads it automatically on first use.

To use a specific model, pass `--model path/to/model.pt` to the CLI or set the model path in the web UI.

To train the PPE model yourself, see [Training a Custom PPE Model](#training-a-custom-ppe-model).

---

## Quick Demo

Generate the demo video and localized zone file:

```bash
python scripts/make_demo_assets.py
```

Run image inference on a real construction scene (person + helmet + vest + goggles + zone intrusion):

```bash
utility-safety-ai infer-image \
  --source examples/sample_images/construction_zone_01.jpg \
  --zones examples/zones_construction_zone_01.yaml \
  --output outputs/demo-ppe/construction_zone_01 \
  --blur-faces
```

Run a full PPE detection demo that also shows gloves and boots:

```bash
utility-safety-ai infer-image \
  --source examples/sample_images/construction_site_ppe_01.jpg \
  --zones examples/zones_construction_site_ppe_01.yaml \
  --output outputs/demo-ppe/construction_site_ppe_01 \
  --blur-faces
```

Or run the no-zone full-PPE reference image:

```bash
utility-safety-ai infer-image \
  --source examples/sample_images/construction_worker_gloves_01.jpg \
  --output outputs/demo-ppe/gloves_demo \
  --blur-faces
```

Run video inference on the generated panning clip:

```bash
utility-safety-ai infer-video \
  --source examples/sample_videos/construction_site_pan.mp4 \
  --zones examples/zones_construction_zone_01.yaml \
  --output outputs/demo-ppe-video \
  --blur-faces
```

Run a full-PPE video demo (person, helmet, vest, gloves, boots, goggles):

```bash
utility-safety-ai infer-video \
  --source examples/sample_videos/construction_ppe_pan.mp4 \
  --output outputs/demo-ppe-full-ppe-video \
  --blur-faces
```

Run live camera / RTSP / webcam inference:

```bash
# Webcam (device 0) for 10 seconds
utility-safety-ai infer-camera \
  --source 0 \
  --zones examples/zones_construction_zone_01.yaml \
  --output outputs/demo-camera \
  --blur-faces \
  --duration 10

# RTSP stream
utility-safety-ai infer-camera \
  --source rtsp://user:pass@camera.local/stream \
  --output outputs/rtsp-demo \
  --duration 30
```

Validate a trained model and produce metrics, confusion matrix, and PR/F1 curves:

```bash
make validate
# or
python scripts/validate_ppe_model.py --model models/ppe_yolo11n.pt --output outputs/validation
```

Benchmark FPS on CPU and available accelerators:

```bash
make benchmark
# or
python scripts/benchmark.py --model models/ppe_yolo11n.pt --output outputs/benchmark
```

Export a trained model for deployment:

```bash
python scripts/export_model.py --model models/ppe_yolo11n.pt --format onnx --output outputs/export
```

Export a CSV report:

```bash
utility-safety-ai export-report \
  --events outputs/demo-ppe/construction_zone_01/events/events.jsonl \
  --output outputs/demo-ppe-report.csv
```

---

## Training a Custom PPE Model

The system supports training on the Ultralytics **Construction-PPE** dataset so it can detect positive PPE classes (`helmet`, `vest`, `gloves`, `boots`, `goggles`) and missing-PPE classes (`no_helmet`, `no_goggle`, `no_gloves`, `no_boots`).

> **Dataset note**: `construction-ppe.yaml` is an Ultralytics dataset alias. If it is not
> auto-downloaded, download the Construction-PPE dataset and point `--data` to its
> `data.yaml`, or run `python scripts/validate_ppe_model.py --data <path>`.

```bash
utility-safety-ai train \
  --data construction-ppe.yaml \
  --model yolo11n.pt \
  --epochs 30 \
  --imgsz 640 \
  --project runs/train_ppe \
  --name ppe_yolo11n_30ep
```

After training, copy the best weights so the CLI and web app use them automatically:

```bash
mkdir -p models
cp runs/train_ppe/ppe_yolo11n_30ep/weights/best.pt models/ppe_yolo11n.pt
```

### Example training results

Models trained for 30 epochs on the Ultralytics Construction-PPE dataset achieved the following validation mAP50 on the 143-image validation split:

| Class     | yolo11n | yolo11s | Notes |
|-----------|---------|---------|-------|
| all       | 0.588   | 0.602   | Aggregate across all classes |
| helmet    | 0.846   | 0.809   | Reliable head protection detection |
| gloves    | 0.810   | 0.804   | Reliable hand protection detection |
| vest      | 0.849   | 0.826   | Reliable high-visibility vest detection |
| boots     | 0.808   | 0.818   | Reliable foot protection detection |
| goggles   | 0.793   | 0.820   | Reliable eye protection detection |
| Person    | 0.905   | 0.911   | Reliable person detection |
| no_helmet | 0.400   | 0.477   | Moderate; depends on viewpoint and lighting |
| no_goggle | 0.177   | 0.217   | Weak; few explicit no-goggle training examples |
| no_gloves | 0.248   | 0.282   | Weak; often confused with occluded hands |
| no_boots  | 0.079   | 0.030   | Very weak; not reliable for alerts in this prototype |

`yolo11n` is the default because it runs roughly 2× faster on CPU while delivering nearly identical positive-PPE quality. `yolo11s` is available as a higher-capacity alternative if you can tolerate the extra latency.

Positive PPE classes (`helmet`, `vest`, `gloves`, `boots`, `goggles`) are detected reliably and are the primary demo focus. Explicit missing-PPE classes (`no_*`) are much harder with the public dataset and are documented honestly; they should not be used as the sole trigger for safety enforcement without further data collection and validation.

Validate the trained model and save per-class metrics:

```bash
make validate
# or
python scripts/validate_ppe_model.py --model models/ppe_yolo11n.pt --output outputs/validation
```

Then run inference without specifying `--model`:

```bash
utility-safety-ai infer-image \
  --source examples/sample_images/construction_zone_01.jpg \
  --zones examples/zones_construction_zone_01.yaml \
  --output outputs/demo-ppe
```

For your own data, create a standard Ultralytics YOLO dataset YAML:

```yaml
path: datasets/my_ppe
train: images/train
val: images/val
nc: 11
names:
  - person
  - helmet
  - vest
  - gloves
  - boots
  - goggles
  - no_helmet
  - no_vest
  - no_gloves
  - no_boots
  - no_goggles
```

---

## Web Demo

```bash
streamlit run app.py
```

The demo supports:

- Image and video upload
- Webcam snapshot
- RTSP / live camera URL (processes a configurable number of frames)
- Model, confidence, IoU, device selection
- Privacy blur toggle
- Preset or custom restricted-zone YAML editor
- Annotated result/video preview
- Events table with risk badges
- Detections table and class distribution chart
- Per-person PPE compliance table
- Summary metrics
- Downloadable CSV, JSONL, and ZIP reports

---

## Project Structure

```text
utility-site-safety-ai-system/
├── app.py                          # Streamlit web demo
├── environment.yml                 # Conda environment spec
├── requirements.txt                # Pip dependencies
├── pyproject.toml                  # Package metadata + entry points
├── Makefile                        # Install / test / demo helpers
├── README.md
├── AGENTS.md                       # Agent implementation spec
├── examples/                       # CC0 sample images, videos, zones
│   ├── sample_images/
│   ├── sample_videos/
│   └── zones_*.yaml
├── datasets/                       # Downloaded PPE datasets (gitignored)
├── scripts/                        # Demo assets, validation, benchmark, export
├── src/utility_safety_ai/          # Main package
│   ├── cli.py
│   ├── compliance/                 # Person-PPE association + compliance reports
│   ├── detection/
│   ├── events/
│   ├── pipelines/
│   ├── privacy/
│   ├── rules/
│   ├── tracking/
│   ├── training/
│   ├── utils/
│   ├── visualization/
│   └── zones/
├── tests/                          # pytest suite
└── outputs/                        # Generated results (gitignored)
```

---

## Output Layout

After inference you will find:

```text
outputs/<run>/
├── images/            # Annotated input images
├── videos/            # Annotated output videos
├── events/
│   ├── events.jsonl   # Structured safety-event log
│   ├── events.csv     # Safety-event spreadsheet
│   ├── detections.jsonl  # Every detected object
│   ├── detections.csv    # Detection spreadsheet
│   ├── compliance.jsonl  # Per-person PPE compliance
│   ├── compliance.csv    # Per-person PPE compliance (CSV)
│   ├── summary.json   # Aggregated counts
│   └── summary.csv    # Aggregated counts (CSV)
└── snapshots/         # Cropped evidence per event
```

---

## Tests

```bash
pytest -q
```

Optional linting:

```bash
ruff check .
```

---

## Resume Bullets

- Built an open-source computer vision safety monitoring system for utility and construction scenarios, supporting PPE compliance detection, restricted-zone intrusion alerts, event logging, privacy-preserving face blurring, and web-based inspection reports.
- Designed a modular safety rule engine that converts YOLO detections and restricted-zone geometry into risk-ranked safety events, with event de-duplication, CSV/JSONL reporting, and aggregated run summaries.
- Delivered CPU-first inference pipelines, a Streamlit demo with preset hazard zones, and reproducible Conda packaging suitable for open-source portfolio and technical demo videos.

---

## Model Capabilities and Limitations

### What the default model can do

- If `models/ppe_yolo11n.pt` (or `models/ppe_yolo11s.pt`) is present, the system can detect **person**, **helmet**, **vest**, **gloves**, **boots**, and **goggles** as well as explicit missing-PPE classes (`no_helmet`, `no_goggle`, `no_gloves`, `no_boots`).
- If no PPE model is found, the default `yolo11n.pt` (COCO pretrained) reliably detects **people** and supports **restricted-zone intrusion** alerts.

### What a custom PPE model can do

- A model fine-tuned on the Ultralytics Construction-PPE dataset learns site-specific positive PPE classes and missing-PPE classes.
- The rule engine emits PPE violation events and escalates risk when, for example, a worker without a helmet enters a high-risk zone.

### Limitations

- **Not production-certified.** This is a research and engineering prototype; human safety officers remain essential.
- **Model performance depends on dataset quality.** The provided training command is a starting point; real deployments need site-specific data and validation. PPE classes such as `vest`, `gloves`, and `boots` may be rare or low-confidence until the model is trained on representative data.
- **Missing-PPE events are only emitted for explicit `no_*` detections or geometry-based person-PPE association.** The system does **not** infer a missing helmet from the absence of a `helmet` box, because that produces too many false positives. Person-level compliance reports mark unobserved items as `unknown` rather than `no`.
- **Person-PPE association** depends on bounding-box overlap; heavily occluded or distant items may be mis-assigned or marked `unknown`.
- **Restricted-zone accuracy** depends on camera perspective, calibration, and correct polygon configuration.
- **Face blurring is a privacy aid**, not a guarantee of full anonymisation.
- **False positives and negatives** are possible; always review event evidence before acting.
- **Industrial deployment gaps** include multi-camera tracking, edge-device optimization, real-time streaming latency, robustness to night/rain, role-based alert routing, audit logging retention, and regulatory certification.

---

## License

MIT License — see [LICENSE](LICENSE).

Sample images `construction_zone_01.jpg`, `electrical_engineer_01.jpg`, and `solar_farm_01.jpg` are sourced from [PxHere](https://pxhere.com) under CC0 / public domain.
`construction_site_ppe_01.jpg` and `construction_worker_gloves_01.jpg` are samples from the Ultralytics Construction-PPE dataset and are included for full-PPE detection demonstration.
`construction_ppe_pan.mp4` is generated from `construction_site_ppe_01.jpg` for video-demo purposes.
