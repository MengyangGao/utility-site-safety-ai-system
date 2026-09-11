# Development

Python 3.10–3.12 are supported; 3.11 is the reference runtime. Dependencies live in `pyproject.toml`
and the committed `uv.lock`. Runtime, API/MQTT extras and developer tools are kept explicit.
Two old direct-only constraints files were replaced by the complete lockfile.

```bash
uv sync --locked --extra dev
uv run --extra dev pytest --cov=utility_safety_ai --cov-fail-under=75
uv run --extra dev ruff check .
uv run --extra dev ruff format --check .
uv run --extra dev mypy src/utility_safety_ai
uv run utility-safety-ai audit-provenance
uv build
```

Tests are under `tests/`, outside the installed source package. Unit tests use deterministic
fake detections; they do not download weights or access a camera. Wheel checks also verify that
the bundled model and examples are available outside the checkout.

## Real optional integration checks

These start temporary loopback services only. Install FFmpeg, MediaMTX and Mosquitto using your
system package manager. They are test tools, not services started automatically by this package.

```bash
# Publisher interruption, reconnect and real bundled PPE inference
uv run --extra dev python tools/smoke_rtsp.py

# Steady transport must complete without reconnects
uv run --extra dev python tools/smoke_rtsp.py --steady --capture-only --output outputs/rtsp-steady

# Actual HTTP 503 retry, HMAC, idempotency and MQTT QoS 1/subscriber delivery
uv run --extra dev python tools/smoke_integrations.py

# Actual dashboard, model inference, downloads and H.264 browser playback
uv sync --locked --extra dev --group browser
uv run --extra dev --group browser playwright install chromium
uv run --extra dev --group browser python tools/smoke_web.py
```

The browser test uses a temporary session and clears integration configuration inherited from
the calling environment. Generated screenshots and logs go into `outputs/` by default. When
promoting a screenshot into `docs/assets`, register its derivation and SHA-256 in the provenance
manifest; retain historical model-evaluation files without rewriting their measurements.

## Accelerators

The uv reference configuration selects CPU PyTorch wheels on Linux/Windows and the normal
MPS-capable wheel on macOS. This keeps the demo/CI install from pulling CUDA runtime packages.
For CUDA or another accelerator, create a separate environment using the
[official PyTorch/uv guidance](https://docs.astral.sh/uv/guides/integration/pytorch/), install this
project with pip, and explicitly choose the device. CUDA and TensorRT were not validated during
this modernization. Model export formats depend on the available exporter and runtime.

## Change policy

Keep detection, policy evaluation, persistence and transports independently testable. New rules
must distinguish uncertain observations from explicit evidence and test temporal behaviour.
Add regressions for stream gaps, partial writes, retries and privacy before changing those paths.
Do not infer missing PPE from the absence of a positive box. Do not publish measured quality
without a frozen checkpoint, representative data and per-class results.
