# Development guide

## Setup

```bash
conda env create -f config/environment.yml
conda activate utility-safety-ai
pip install -e ".[dev]"
```

`pyproject.toml` is the only dependency source of truth. `requirements.txt` and the Conda file are
thin convenience entry points. A known-good Python 3.11 reference is available at
`config/constraints-py311.txt`.

## Quality gate

```bash
pip check
pytest -q --cov=utility_safety_ai --cov-report=term-missing --cov-fail-under=70
ruff check .
mypy src/utility_safety_ai
python -m py_compile src/utility_safety_ai/web/app.py
utility-safety-ai audit-provenance
python -m build
```

The default suite must stay offline. Model downloads and full inference belong in explicit release
checks. Every behavior change should include focused tests for success and failure paths.

## Repository layout

```text
config/                       environment and reference constraints
docs/                         product, model/data, legal, and verification records
examples/                     licensed sample media and normalized zones
models/                       ignored weights plus public registry
scripts/                      validation, benchmark, export, and asset tooling
src/utility_safety_ai/        package, pipelines, CLI, and Web console
tests/                        offline unit and end-to-end tests
```

The root intentionally contains only `.gitignore`, `LICENSE`, `README.md`, `pyproject.toml`, and
`requirements.txt`.

## Engineering rules

- Keep CPU operation as the supported default; acceleration is optional.
- Never infer missing PPE from absence of a positive box.
- Keep track IDs temporary and local to one run.
- Separate active findings from newly persisted events.
- Apply privacy processing before saving media.
- Fail clearly on invalid zones, corrupt media, or output errors.
- Redact credentials and avoid unnecessary personal data.
- Do not load untrusted `.pt` checkpoints.
- Preserve completed run history unless overwrite is explicit.

## Contribution checklist

- Add or update tests and documentation.
- Run the complete quality gate.
- Keep capability and performance claims tied to retained evidence.
- Do not commit secrets, private footage, datasets, or weights.
- Register every third-party/generated artifact in `docs/legal/provenance.yaml`.
- Use only an approved open-source/public-domain license and exact SHA-256.
- Report security issues privately through GitHub Security Advisories.

For model contributions, include base model/license, checkpoint hash, dataset version/license/split,
training environment, per-class metrics, limitations, and redistribution status.

## Release demo

```bash
utility-safety-ai fetch-model --model yolo11n.pt --output models
utility-safety-ai audit-provenance
utility-safety-ai infer-image \
  --source examples/sample_images/construction_site_ppe_01.jpg \
  --model models/yolo11n.pt \
  --zones examples/zones_construction_site_ppe_01.yaml \
  --output outputs/release \
  --run-id portfolio-demo \
  --overwrite \
  --blur-faces
utility-safety-ai web
```

Show the model class list, privacy default, normalized zone, annotated result, event evidence,
quality panel, manifest hashes, and limitations. Never attach PPE metrics to a different checkpoint.
