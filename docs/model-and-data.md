# Models and training

## Ready-to-use models

| Model | Size | Classes / capabilities |
|---|---:|---|
| `models/ppe_yolo11n.pt` | 5.2 MB | Person, helmet, vest, gloves, boots, goggles, explicit missing-PPE labels, zones |
| `models/yolo11n.pt` | 5.4 MB | COCO person detection and zones |

The application selects `ppe_yolo11n.pt` automatically. Choose another model with `--model` or the
**Verified model profile** control in the Web console.

## PPE class coverage

The bundled PPE checkpoint was fine-tuned from Ultralytics YOLO11n on
[Construction-PPE 1.0.0](https://docs.ultralytics.com/datasets/detect/construction-ppe/):

```text
helmet, gloves, vest, boots, goggles
person
no_helmet, no_goggle, no_gloves, no_boots
```

Construction-PPE contains 1,416 images across train, validation, and test splits. It does not include
a `no_vest` class, so the application does not infer a missing vest merely because no vest box was
detected. Unsupported or uncertain PPE states remain `unknown`.

## Evaluate the included PPE model

```bash
python -m utility_safety_ai.tools.validate_ppe_model \
  --model models/ppe_yolo11n.pt \
  --data construction-ppe.yaml \
  --output outputs/validation/ppe_yolo11n
```

The retained validation result and plots are in
[`model-evaluation/ppe_yolo11n-v1`](model-evaluation/ppe_yolo11n-v1/README.md).

## Fine-tune a model

Ultralytics downloads Construction-PPE automatically when the dataset name is used:

```bash
utility-safety-ai train \
  --data construction-ppe.yaml \
  --model models/yolo11n.pt \
  --epochs 30 \
  --imgsz 640
```

For a custom dataset, provide an Ultralytics-format YAML file with `train`, `val`, and `names`
fields. Copy the selected `best.pt` checkpoint into `models/` and pass its path with `--model`.

## Export for deployment

```bash
utility-safety-ai export-model \
  --model models/ppe_yolo11n.pt \
  --format onnx \
  --output outputs/export
```

Supported export targets depend on the local Ultralytics installation and include ONNX,
TorchScript, OpenVINO, CoreML, TensorRT, TFLite, and NCNN.

## Improve detection quality

- Collect consented examples from the intended camera height, lighting, weather, and PPE styles.
- Split data by site, camera, and recording session instead of adjacent frames.
- Add more real examples of rare missing-PPE classes.
- Compare per-class precision and recall, not only aggregate metrics.
- Review end-to-end alerts after tracking, zone logic, and PPE association.
