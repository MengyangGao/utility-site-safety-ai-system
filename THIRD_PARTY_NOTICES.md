# Third-Party Notices and License Boundaries

This file is a practical inventory, not legal advice. The root `LICENSE` applies to original project code only to the extent that code can be licensed separately. It does not replace or weaken the licenses of dependencies, models, datasets, fonts, codecs, media, or generated derivatives.

## Critical Ultralytics boundary

The project imports and depends on the `ultralytics` package. Ultralytics' official licensing materials describe two principal routes:

- AGPL-3.0 for qualifying open-source use; and
- a commercial Enterprise license for use cases that do not meet the applicable open-source obligations.

Ultralytics also states that its trained YOLO models are covered by AGPL-3.0 by default, with Enterprise terms available. Review the current [Ultralytics license page](https://www.ultralytics.com/license), [Ultralytics repository](https://github.com/ultralytics/ultralytics), and the exact license bundled with the installed distribution before use or redistribution.

The presence of an MIT license in this repository is **not** a statement that the combined Ultralytics-based application, downloaded YOLO weights, or fine-tuned derivatives can be used under MIT alone. Proprietary, internal-company, SaaS, embedded, commercial, or closed deployments require a specific license review; replacing the detector backend may be appropriate if those terms do not fit.

## Models

| Item | Obtained how | Upstream terms | Repository policy |
|---|---|---|---|
| `yolo11n.pt` | Downloaded by Ultralytics or `utility-safety-ai fetch-model` | Ultralytics identifies YOLO11 software/models with AGPL-3.0 and Enterprise options | Not committed by default; store locally with generated SHA-256 metadata |
| `models/ppe_yolo11n.pt` / `ppe_yolo11s.pt` | User-provided or locally fine-tuned | Depends on base model, training framework, dataset, and any additional agreement | Do not redistribute until all upstream/model/dataset terms are reviewed |
| Exported ONNX/OpenVINO/CoreML/TensorRT/etc. | Derived from a checkpoint | Exporting format does not erase source-model obligations | Retain source hash, export metadata, notices, and deployment-runtime terms |

Official YOLO11 documentation: <https://docs.ultralytics.com/models/yolo11/>

## Dataset

The optional Construction-PPE workflow uses the Ultralytics dataset alias `construction-ppe.yaml`. Its official dataset page identifies 1,416 images, 11 classes, and an AGPL-3.0 license:

<https://docs.ultralytics.com/datasets/detect/construction-ppe/>

The repository contains its configuration/license material under `datasets/`, not a grant to relicense the dataset. Fine-tuned weights may inherit obligations from both the base model and dataset. Review [`docs/dataset-card.md`](docs/dataset-card.md) before training or redistribution.

## Example media

The authoritative per-file source, SHA-256, author, license claim, and derivation status is [`examples/assets.yaml`](examples/assets.yaml).

Current categories include:

- verified CC0 media;
- verified Pexels-licensed media;
- samples from the AGPL-3.0 Construction-PPE dataset;
- generated videos derived from those still images.

CC0 does not remove privacy, publicity, trademark, or endorsement considerations. The [Creative Commons CC0 deed](https://creativecommons.org/publicdomain/zero/1.0/) expressly notes such separate rights. The [Pexels License](https://www.pexels.com/license/) also includes restrictions beyond free use. Do not imply that depicted people or organizations endorse this project.

Generated panning videos inherit the source image's relevant terms; generation does not create a clean license boundary.

## Python dependencies

Runtime and development dependencies are declared in `pyproject.toml` and `requirements.txt`. Major packages include:

| Package | Typical upstream license in the supported line | Source |
|---|---|---|
| Ultralytics | AGPL-3.0 or separate Enterprise terms | <https://github.com/ultralytics/ultralytics> |
| PyTorch | BSD-style and bundled component licenses | <https://github.com/pytorch/pytorch> |
| OpenCV / `opencv-python` | Apache-2.0 plus bundled component notices | <https://github.com/opencv/opencv-python> |
| NumPy | BSD-3-Clause plus bundled component notices | <https://github.com/numpy/numpy> |
| pandas | BSD-3-Clause | <https://github.com/pandas-dev/pandas> |
| Pillow | HPND-style Pillow license | <https://github.com/python-pillow/Pillow> |
| Streamlit | Apache-2.0 | <https://github.com/streamlit/streamlit> |
| Click | BSD-3-Clause | <https://github.com/pallets/click> |
| PyYAML | MIT | <https://github.com/yaml/pyyaml> |
| `lap` | BSD-2-Clause | <https://github.com/gatagat/lap> |

This summary is not a substitute for the license metadata and notice files in the exact installed wheel, native library, or transitive dependency. Binary NumPy, OpenCV, PyTorch, codec, and accelerator distributions may bundle additional components.

## Release checklist

Before publishing a binary, container, hosted service, model, dataset subset, or media bundle:

1. Freeze the exact dependency and model versions.
2. Generate an SBOM/license report from the final artifact.
3. Retain required copyright and notice files.
4. Verify every example/media source against `examples/assets.yaml`.
5. Re-verify every media URL, hash, and applicable term for the release.
6. Review model and dataset redistribution rights.
7. Review AGPL/network-use obligations or obtain appropriate commercial terms.
8. Review exported-runtime and codec licenses for the target platform.
9. Obtain qualified legal advice for commercial or organizational deployment.
