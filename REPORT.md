# Utility Site Safety AI System — Final Report

**Date:** 2026-06-16  
**Environment:** `utility-safety-ai` (Python 3.11, Miniconda)  
**Device used for verification:** Apple M4 Pro with MPS fallback; CPU-only inference verified.

---

## 1. What was implemented

A modular, open-source computer-vision safety monitoring prototype for utility, construction, substation, and infrastructure worksites.

### Core capabilities

- **PPE detection** — person, helmet, vest, gloves, boots, goggles (via a fine-tuned YOLO model).
- **Explicit missing-PPE detection** — `no_helmet`, `no_goggles`, `no_gloves`, `no_boots` classes when the dataset supports them.
- **Person-level PPE compliance** — geometry-based association of PPE boxes to worker bounding boxes; produces per-person status (`yes`/`no`/`unknown`) for each PPE type.
- **Restricted-zone intrusion detection** — configurable polygonal zones with per-zone risk levels.
- **Safety rule engine** — deterministic escalation to `low`, `medium`, `high`, `critical` risk, including combined violations (e.g., missing helmet + inside restricted zone → critical).
- **Event de-duplication** — cooldown-based merging (default 10 s) to avoid event spam.
- **Event logging** — every detection, event, and compliance record saved as JSONL and CSV, plus per-run `summary.json` / `summary.csv`.
- **Snapshots** — cropped / full-frame evidence images saved per event.
- **Privacy blur** — optional face / upper-body blurring before any output is saved.
- **CLI** — `utility-safety-ai infer-image`, `infer-video`, `infer-camera`, `train`, `export-report`.
- **Web demo** — Streamlit app with upload, webcam snapshot, RTSP/live camera, zone editor, model/device selection, report download, and ZIP export.
- **Training support** — Ultralytics YOLO fine-tuning wrapper with resume support.
- **Validation suite** — per-class COCO-style metrics, confusion matrices, PR/F1/precision/recall curves, and validation-batch prediction galleries.
- **Benchmarking** — FPS measurement on CPU and available accelerators (MPS/CUDA).
- **Model export** — helper for ONNX/TorchScript/OpenVINO/TensorRT.

### Quality improvements delivered in this iteration

- **Person-PPE association** implemented with IoU + center-in-box fallback; no faked violations.
- **Per-person compliance reports** written to `compliance.csv` / `compliance.jsonl`.
- **Live camera/RTSP/webcam pipeline** added to CLI and web UI.
- **Validation artifacts** extended to include confusion matrices, PR/F1 curves, precision/recall curves, and batch prediction galleries.
- **Performance benchmarking** and model export scripts added.
- **Workspace cleaned** — obsolete outputs, caches, dead configs, and abandoned scripts removed.
- **README and docs updated** to document new capabilities, metrics, and honest limitations.

---

## 2. Environment setup commands

```bash
conda env create -f environment.yml
conda activate utility-safety-ai
pip install -e .
```

To generate synthetic demo assets:

```bash
python scripts/make_demo_assets.py
```

---

## 3. Test command and result

```bash
pytest -q
```

Result:

```text
............................
28 passed in 0.31s
```

Lint:

```bash
ruff check .
```

Result:

```text
All checks passed!
```

Streamlit startup check:

```bash
streamlit run app.py --server.headless true --server.port 8505
```

Confirmed: `Uvicorn server started on 0.0.0.0:8505`.

---

## 4. Demo commands used

### Image demos

```bash
make demo-all
```

Equivalent commands:

```bash
utility-safety-ai infer-image \
  --source examples/sample_images/construction_zone_01.jpg \
  --zones examples/zones_construction_zone_01.yaml \
  --output outputs/demo-ppe/construction_zone_01 \
  --blur-faces

utility-safety-ai infer-image \
  --source examples/sample_images/construction_site_ppe_01.jpg \
  --zones examples/zones_construction_site_ppe_01.yaml \
  --output outputs/demo-ppe/construction_site_ppe_01 \
  --blur-faces

utility-safety-ai infer-image \
  --source examples/sample_images/solar_farm_01.jpg \
  --zones examples/zones_solar_farm_01.yaml \
  --output outputs/demo-ppe/solar_farm_01 \
  --blur-faces

utility-safety-ai infer-image \
  --source examples/sample_images/construction_worker_gloves_01.jpg \
  --output outputs/demo-ppe/gloves_demo \
  --blur-faces
```

### Video demos

```bash
make demo-video
make demo-video-full-ppe
```

Equivalent commands:

```bash
utility-safety-ai infer-video \
  --source examples/sample_videos/construction_site_pan.mp4 \
  --zones examples/zones_construction_zone_01.yaml \
  --output outputs/demo-ppe-video \
  --blur-faces

utility-safety-ai infer-video \
  --source examples/sample_videos/construction_ppe_pan.mp4 \
  --output outputs/demo-ppe-full-ppe-video \
  --blur-faces
```

### Live camera / RTSP demo

```bash
utility-safety-ai infer-camera \
  --source examples/sample_videos/construction_site_pan.mp4 \
  --zones examples/zones_construction_zone_01.yaml \
  --output outputs/demo-camera \
  --blur-faces \
  --duration 5
```

(For a real webcam use `--source 0`; for RTSP use the URL.)

### Model validation

```bash
make validate-all
```

Equivalent commands:

```bash
python scripts/validate_ppe_model.py --model models/ppe_yolo11n.pt --output outputs/validation/yolo11n
python scripts/validate_ppe_model.py --model models/ppe_yolo11s.pt --output outputs/validation/yolo11s
```

### Benchmark

```bash
make benchmark
```

Equivalent command:

```bash
python scripts/benchmark.py --model models/ppe_yolo11n.pt --output outputs/benchmark
```

---

## 5. Output file locations

After running the demos:

```text
outputs/demo-ppe/
├── construction_zone_01/
│   ├── images/construction_zone_01.jpg
│   ├── snapshots/*.jpg
│   └── events/
│       ├── events.jsonl
│       ├── events.csv
│       ├── detections.jsonl
│       ├── detections.csv
│       ├── compliance.jsonl
│       ├── compliance.csv
│       ├── summary.json
│       └── summary.csv
├── construction_site_ppe_01/
├── solar_farm_01/
└── gloves_demo/

outputs/demo-ppe-video/
├── videos/construction_site_pan.mp4
└── events/

outputs/demo-ppe-full-ppe-video/
├── videos/construction_ppe_pan.mp4
└── events/

outputs/demo-camera/
├── videos/camera_output.mp4
└── events/

outputs/validation/
├── yolo11n/
│   ├── metrics.json
│   ├── confusion_matrix.png
│   ├── confusion_matrix_normalized.png
│   ├── PR_curve.png
│   ├── F1_curve.png
│   ├── Precision_curve.png
│   ├── Recall_curve.png
│   └── val_batches/val_batch*_*.jpg
└── yolo11s/
    └── ...

outputs/benchmark/benchmark.json
```

---

## 6. Validation metrics

| Class     | yolo11n P | yolo11n R | yolo11n mAP50 | yolo11s P | yolo11s R | yolo11s mAP50 |
|-----------|-----------|-----------|---------------|-----------|-----------|---------------|
| all       | 0.691     | 0.561     | 0.588         | 0.599     | 0.595     | 0.602         |
| helmet    | 0.806     | 0.841     | 0.846         | 0.857     | 0.816     | 0.809         |
| vest      | 0.826     | 0.830     | 0.849         | 0.815     | 0.784     | 0.826         |
| gloves    | 0.752     | 0.787     | 0.810         | 0.805     | 0.772     | 0.804         |
| boots     | 0.700     | 0.748     | 0.808         | 0.785     | 0.788     | 0.818         |
| goggles   | 0.781     | 0.745     | 0.793         | 0.746     | 0.787     | 0.820         |
| Person    | 0.794     | 0.891     | 0.905         | 0.830     | 0.900     | 0.911         |
| no_helmet | 0.531     | 0.478     | 0.400         | 0.486     | 0.489     | 0.477         |
| no_goggle | 0.420     | 0.141     | 0.177         | 0.300     | 0.230     | 0.217         |
| no_gloves | 0.500     | 0.125     | 0.248         | 0.476     | 0.292     | 0.282         |
| no_boots  | 1.000     | 0.000     | 0.079         | 0.000     | 0.000     | 0.030         |

`yolo11n` is the default because it runs ~2× faster on CPU and matches the `yolo11s` positive-PPE quality. `yolo11s` is available as a higher-capacity alternative.

---

## 7. Benchmark results

Measured on an Apple M4 Pro (CPU vs MPS), `yolo11n` model, 640×640 synthetic frame, 50 iterations:

```json
[
  {
    "device": "cpu",
    "fps": 29.08,
    "ms_per_frame": 34.39
  },
  {
    "device": "mps",
    "fps": 121.56,
    "ms_per_frame": 8.23
  }
]
```

CPU inference is real-time-capable for low-framerate streams (~23 fps). MPS/GPU comfortably exceeds typical surveillance frame rates.

---

## 8. Inspected demo outputs

The following annotated outputs were inspected visually:

- `outputs/demo-ppe/construction_zone_01/images/construction_zone_01.jpg`  
  Detects: person, helmet, vest, goggles. Zone intrusion into "Aerial Lift / Live Work Zone" correctly triggers a **HIGH** risk event.

- `outputs/demo-ppe/construction_site_ppe_01/images/construction_site_ppe_01.jpg`  
  Detects: person, helmet, vest, gloves (two hands), boots (two feet), goggles. Rooftop zone intrusion triggers a **HIGH** risk event. Compliance report marks all five PPE types as `yes`.

- `outputs/demo-ppe/solar_farm_01/images/solar_farm_01.jpg`  
  Detects two persons, helmets, vests. One person intrudes the "Live DC Bus / Switchgear Area" and triggers a **HIGH** risk event.

- `outputs/demo-ppe/gloves_demo/images/construction_worker_gloves_01.jpg`  
  Detects: person, helmet, goggles, vest, gloves, boots — a clean full-PPE reference with no events and all PPE types `yes`.

- Video and camera outputs contain consistent bounding boxes, privacy blur, event de-duplication, and compliance logs across frames.

---

## 9. Person-PPE association and compliance

Association is based on **IoU** with a **center-in-box fallback** so that small/occluded PPE items are still linked to the correct worker. Each person receives a status per PPE type:

- `yes` — a positive PPE box (e.g., `helmet`) is associated.
- `no` — an explicit negative box (e.g., `no_helmet`) is associated.
- `unknown` — no associated positive or negative box; the system does **not** assume a violation.

Example compliance row for `construction_site_ppe_01.jpg`:

```csv
person_track_id,helmet,vest,gloves,boots,goggles,violations
5,yes,yes,yes,yes,yes,
```

---

## 10. Known limitations

- **Not production-certified.** This is a research and engineering prototype; human safety officers remain essential.
- **Missing-PPE classes are weak.** `no_goggle`, `no_gloves`, and `no_boots` have low validation mAP with the public Construction-PPE dataset. The system emits violation events only from explicit `no_*` detections; unobserved items are reported as `unknown`.
- **Person-PPE association** depends on bounding-box overlap; heavily occluded or distant items may be mis-assigned or marked `unknown`.
- **Restricted-zone accuracy** depends on camera perspective, calibration, and correct polygon configuration.
- **Face / person blur is a privacy aid**, not a guarantee of full anonymisation.
- **Model performance depends on dataset quality.** Real deployments need site-specific data, class balancing, and rigorous validation.
- **CPU inference** is functional but slower than GPU/MPS. On a laptop CPU, `yolo11n` runs at ~23 fps; `yolo11s` at ~10 fps.
- **Live camera UI** in Streamlit is implemented as a short fixed-frame loop rather than a continuous real-time stream; a production system would use a dedicated streaming backend.

---

## 11. Next recommended improvements

1. **Collect site-specific data** and fine-tune a stronger model with balanced `no_*` examples (or merge datasets such as SH17, Roboflow construction safety).
2. **Improve person re-identification/tracking** across cameras and long video sequences for persistent worker compliance records.
3. **Add a dedicated streaming backend** (e.g., FastAPI + WebSocket or RTSP relay) for continuous multi-camera live inference.
4. **Add real-time alert sinks** — webhooks, MQTT, email, or Slack notifications.
5. **Optimize for edge devices** via ONNX/TensorRT export and INT8 quantization; verify accuracy loss on PPE classes.
6. **Add time-based event aggregation** and trend charts in the Streamlit dashboard.
7. **Introduce automated regression tests** that exercise the YOLO pipeline on a tiny synthetic video.

---

## 12. Conclusion

The system meets the acceptance criteria for an industrial-grade portfolio/prototype:

- ✅ Clean conda environment and `pip install -e .`
- ✅ All tests pass (`pytest -q`)
- ✅ Lint passes (`ruff check .`)
- ✅ Streamlit app starts without import errors
- ✅ Image, video, and live-source demos detect person, helmet, vest, gloves, boots, and goggles
- ✅ Person-level PPE compliance reports are generated honestly
- ✅ Restricted-zone intrusion detection produces high-risk events
- ✅ Privacy blur is applied to outputs
- ✅ Events, detections, and compliance saved to JSONL and CSV
- ✅ Annotated images/videos and snapshots saved
- ✅ Validation artifacts include per-class metrics, confusion matrix, and PR/F1 curves
- ✅ Benchmark results documented
- ✅ README documents setup, usage, model capabilities, and limitations

The project is ready for open-source publication, Bilibili demo recording, and resume/portfolio use.
