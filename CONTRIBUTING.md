# Contributing

Thank you for helping improve Utility Site Safety AI. Contributions are welcome when they preserve the project's core principle: safety findings must be explainable, auditable, privacy-aware, and honest about model capability.

## Before opening an issue

- Search existing issues and the [security policy](SECURITY.md).
- Do not attach identifiable worksite footage, RTSP credentials, private model weights, or proprietary datasets to a public issue.
- For a bug, include the command, platform, Python version, sanitized traceback, model class list, and a minimal reproducible input when its license permits sharing.
- For a model-quality report, distinguish detector failure, PPE-person association failure, tracking failure, zone geometry, and rule-engine behavior.

## Development setup

```bash
conda env create -f environment.yml
conda activate utility-safety-ai
pip install -e ".[dev]"
```

If the environment already exists:

```bash
make install
```

## Workflow

1. Create a focused branch.
2. Add or update offline tests before changing behavior.
3. Keep changes modular and avoid network/model downloads in the default test suite.
4. Update user-facing documentation and the changelog when behavior changes.
5. Run the quality gate.
6. Open a pull request with scope, evidence, limitations, and any migration notes.

```bash
make verify
python -m build
```

## Engineering standards

- Use `pathlib`, type hints, dataclasses or similarly explicit structures.
- Preserve CPU compatibility; acceleration must remain optional.
- Do not infer a confirmed PPE violation from the absence of a positive detection.
- Keep temporary track IDs local to a run; do not add biometric identity or cross-camera identity features.
- Treat active findings and persisted new events as different concepts.
- Validate external input and fail with actionable errors rather than silently disabling safety logic.
- Apply privacy processing before saving annotated media or evidence.
- Never log raw credentials, tokens, or unnecessary personal information.
- Preserve immutable run history unless the user explicitly chooses overwrite.

## Testing expectations

Every behavior change should include a targeted test. High-value cases include:

- contradictory positive/negative PPE evidence;
- multiple workers and repeated same-type PPE boxes;
- polygon boundaries and normalized scaling;
- cooldown, dwell, stream reset, and blank-frame aging;
- empty logs and failed-run manifests;
- corrupt media and output-write failures;
- privacy blur with associated/unassociated faces;
- CLI exit codes and synthetic image/video pipelines;
- Web helpers without starting a browser or downloading a model.

The core suite must stay offline. Model downloads and full validation belong in explicit integration/release workflows.

## Model and dataset contributions

A model PR must include:

- architecture/base checkpoint and upstream license;
- exact checkpoint SHA-256;
- class names and normalization rules;
- dataset name/version/license and split definition;
- training configuration, package versions, and hardware;
- per-class precision, recall, mAP50, and mAP50-95 artifacts;
- known failure modes, especially negative-PPE classes;
- a clear statement on whether weights can legally be redistributed.

Do not commit large weights or datasets without maintainer approval and verified redistribution rights. Prefer scripts and checksums.

## Example media contributions

Every added or replaced asset must update [`examples/assets.yaml`](examples/assets.yaml) with:

- repository-relative path;
- SHA-256;
- exact source page, author where known, and license;
- derivation/generation details for videos or edited media;
- a provenance status.

Assets with incomplete provenance should not become the primary demo path.

## Pull request checklist

- [ ] Scope is focused and documented.
- [ ] Tests cover success and failure behavior.
- [ ] `pytest` with the configured coverage gate passes.
- [ ] Ruff and mypy pass.
- [ ] No secrets, personal data, weights, datasets, or untracked media were added accidentally.
- [ ] CLI/Web/docs agree on commands and output paths.
- [ ] License and provenance were reviewed for every new dependency, model, dataset, and asset.
- [ ] Safety and privacy claims remain appropriately limited.

## Documentation-only changes

Documentation fixes do not require model inference, but commands and file paths must be checked against the current code. Do not paste remembered benchmark or validation values without retained artifacts.
