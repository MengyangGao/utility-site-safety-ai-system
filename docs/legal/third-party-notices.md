# Third-party notices

This inventory records the project's open-source license boundary. It is not legal advice and does
not replace license texts shipped by exact package, binary, model, dataset, codec, or accelerator
distributions.

## Project license

Original project code and documentation are distributed under **GNU AGPL-3.0-only**. The root
[`LICENSE`](../../LICENSE) contains the complete text. The Streamlit interface links users to the
source and license.

The reviewed Git history contains commits authored as `Mengyang Gao` and `AI Agent`, with no
submodules, Git LFS objects, `Co-authored-by`, or `Signed-off-by` trailers. Publication under this
license assumes the repository owner controls or has permission to license those contributions. If
code arrived outside the recorded history or from another contributor, obtain their permission or
remove/replace it before release.

A repository-wide source scan found no embedded third-party copyright header, copied/adapted-source
marker, vendored code directory, or submodule in the application, scripts, and tests. This is a
traceability check, not a legal determination of originality.

The license was selected to align the repository with its integrated Ultralytics runtime instead of
suggesting that the combined application is available under MIT alone. Commercial use remains
possible under AGPL compliance; organizations seeking a proprietary Ultralytics deployment should
review Ultralytics' separate licensing options.

## Detector software and models

| Component | Source | Declared license |
|---|---|---|
| Ultralytics runtime | <https://github.com/ultralytics/ultralytics> | AGPL-3.0 or separate Enterprise terms |
| YOLO11 checkpoints | <https://docs.ultralytics.com/models/yolo11/> | Ultralytics AGPL-3.0 / Enterprise boundary |
| Local PPE derivatives | User-trained from YOLO11 and Construction-PPE | AGPL-3.0-only for this documented workflow |

Weights are excluded from Git. Exporting a checkpoint to ONNX, OpenVINO, CoreML, TensorRT, or
another format does not erase source-model, dataset, or runtime obligations.

## Dataset and evaluation

The optional training workflow uses [Ultralytics Construction-PPE](https://docs.ultralytics.com/datasets/detect/construction-ppe/),
documented upstream as AGPL-3.0. The repository contains only its configuration, license, two exact
sample images, one derived demo video, and hash-addressed evaluation evidence—not the dataset archive
or model checkpoint.

The prior validation mosaics were removed because their exact constituent filenames were not
retained. Numeric results and aggregate plots remain bound to the recorded dataset archive,
validation split, environment, and local checkpoint hash.

## Media policy

Every committed media/evaluation artifact is registered in
[`provenance.yaml`](provenance.yaml). The automated audit rejects:

- an unregistered audited file;
- a missing or duplicate path;
- SHA-256 drift;
- a license outside the approved `AGPL-3.0-only` / `CC0-1.0` set.

No Pexels or custom stock-license media is committed. A source license does not remove privacy,
publicity, trademark, or endorsement concerns; privacy redaction and human review remain required.

## Direct Python dependencies

| Package | Upstream license family | Source |
|---|---|---|
| Ultralytics | AGPL-3.0 / Enterprise | <https://github.com/ultralytics/ultralytics> |
| PyTorch | BSD-style plus bundled notices | <https://github.com/pytorch/pytorch> |
| OpenCV / opencv-python | Apache-2.0 plus bundled notices | <https://github.com/opencv/opencv-python> |
| NumPy | BSD-3-Clause plus bundled notices | <https://github.com/numpy/numpy> |
| pandas | BSD-3-Clause | <https://github.com/pandas-dev/pandas> |
| Pillow | HPND-style Pillow License | <https://github.com/python-pillow/Pillow> |
| Streamlit | Apache-2.0 | <https://github.com/streamlit/streamlit> |
| Click | BSD-3-Clause | <https://github.com/pallets/click> |
| PyYAML | MIT | <https://github.com/yaml/pyyaml> |
| lap | BSD-2-Clause | <https://github.com/gatagat/lap> |
| build, pytest, pytest-cov, Ruff, mypy, setuptools, wheel | OSI-approved licenses; development/build only | Declared project repositories and installed metadata |

Binary distributions can include further components such as BLAS, codecs, compiler runtimes, and
accelerator libraries. Preserve the notices shipped with the exact release artifact.

## Publication checklist

Before publishing a wheel, container, hosted service, model, dataset subset, or media bundle:

1. run `utility-safety-ai audit-provenance`;
2. freeze exact dependency/model versions and generate an SBOM/license report;
3. retain all required copyright and notice files;
4. verify model/dataset/media hashes and redistribution rights;
5. provide corresponding source and AGPL network-use notices where required;
6. review privacy, publicity, trademark, codec, export, and deployment-specific obligations;
7. obtain qualified legal advice for organizational or commercial release.
