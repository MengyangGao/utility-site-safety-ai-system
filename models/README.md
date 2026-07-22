# Included models

The repository is ready to run immediately after installation.

| Model | Size | Use |
|---|---:|---|
| `ppe_yolo11n.pt` | 5.2 MB | Default PPE, person, and restricted-zone monitoring |
| `yolo11n.pt` | 5.4 MB | General person and restricted-zone monitoring |

The PPE model was fine-tuned from Ultralytics YOLO11n on the
[Construction-PPE dataset](https://docs.ultralytics.com/datasets/detect/construction-ppe/). Its
class map includes worn PPE and explicit missing-PPE labels. The application selects it
automatically; pass `--model models/yolo11n.pt` when only person and zone monitoring is needed.

Model metadata and hashes are available in [`registry.json`](registry.json).
