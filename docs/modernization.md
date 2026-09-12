# Modernization audit — September 2026

Baseline: `b2f2922f01ed1388b4609b696e0358fd25f9a39f`, the existing v2.1
monitoring application. The complete available history, source, tests, configuration, model
metadata and documentation were reviewed before changes. The detector, trained checkpoints,
association rules, zone policies, artifact structure and Streamlit application were retained.

## Problems and changes

| Finding | Change |
|---|---|
| Live capture could block indefinitely or accumulate old frames | FFmpeg open/read timeouts; TCP default; one-frame mailbox; stale rejection; bounded reconnect and shutdown |
| Webhook delivery waited until the entire camera/video run ended | Independent delivery worker, durable outbox, leases, retry backoff, dead letters, HMAC and idempotency |
| Temporary IDs could be reused after reconnect; some state grew indefinitely | Explicit stream segments, bounded cooldown/compliance caches and stable observation IDs |
| Lifecycle transitions were calculated but not retained | Observation lifecycle JSONL and explicit interruption records |
| Operator decisions overwrote a CSV within a completed run | Append-only SQLite review history, exported separately from original hashed evidence |
| Standard OpenCV MP4 output could fail in browsers | Optional FFmpeg H.264 conversion before hashing; explicit fallback status when FFmpeg is missing |
| Wheel smoke checked help text but omitted bundled assets | Models and examples included in wheel; resource resolver and real installed-wheel inference |
| NaN values could pass model gates; absent classes could shift metric rows | Finite range validation and class-ID-aware metric mapping with unmeasured values kept null |
| Duplicate export implementations handled directories incorrectly | One export implementation, directory support and explicit INT8 calibration requirement |
| Unused synthetic-negative generation modified validation data and could use training data as validation | Retired the unused augmentation script; real-data training/evaluation remains available |
| Stale dependency constraints, tests in source package, dead helpers and fragile file-count tests | Complete uv lockfile, CPU reference wheels, `tests/` layout and removed unused code |
| Upstream runtime enabled usage analytics | Process-local usage-event opt-out and offline default for local inference; global user settings remain unchanged |

The guide's invalid `min_dwell_seconds` example was corrected to the actual `dwell_seconds`
field. CSV output now escapes formula-like text while JSONL retains original evidence strings.

## Local validation

The original baseline passed **152 tests, 77.07% coverage**. The modernized suite passed
**188 tests, 81.27% coverage** on reference Python 3.11. The supported Python 3.10 and 3.12
runs also passed 188 tests each; see the [validation summary](validation/local-checks.json). Ruff lint/format, mypy and actionlint passed.

Additional checks used actual runtimes:

- Bundled PPE checkpoint on the included still image: 8 detections, 1 zone event. This is a
  functional example, not an accuracy measurement.
- A fresh wheel environment outside the checkout found its bundled assets and completed real
  CPU inference without downloading a model.
- [Steady loopback RTSP](validation/rtsp-steady.json): 30 processed frames, no reconnects.
- [Interrupted RTSP publisher](validation/rtsp-reconnect.json): real PPE inference, one recovered
  continuity segment, credential-free source, and monotonic timestamp records. Early experiments
  with two-second probe limits caused unnecessary reconnects; the five-second default was retained.
- [Webhook and MQTT](validation/protocols.json): actual local HTTP 503 retry, HMAC verification,
  stable idempotency key, MQTT QoS 1 acknowledgement and subscriber receipt.
- Chromium: real image/video analysis, automatic results navigation, report download, artifact
  hash verification and H.264 playback. Screenshots were captured from this workflow.
- Streamlit AppTest: an edited review appended history without changing the original manifest.
- Runtime privacy check: real inference completed with usage events and the online probe disabled;
  the runtime's settings file was unchanged by the opt-out.
- `pip-audit` reported no known advisories for the audited installed dependencies. The local
  unpublished package itself is not covered by that advisory scan.

Tests and tools use synthetic fixtures or provenance-tracked repository examples. No physical
camera, external alert receiver, paid service, new accuracy benchmark or field deployment was
used. These checks were first performed locally; subsequent remote validation is recorded below.

## Remaining boundaries

The retained model has weak missing-PPE classes and no `no_vest` class. More representative
site/camera data, rare-class validation, adverse-weather footage and physical-camera soak tests
are still needed. The pan video is derived from a still image and is not motion-tracking ground
truth. Re-identification, general dangerous-behaviour recognition and multi-camera orchestration
are outside this demo.

Native capture behaviour depends on the OpenCV backend and camera firmware. A backend that ignores
timeouts requires process isolation under an external supervisor. Long runs retain audit data on
disk and returned events in memory; the default camera duration is bounded to five minutes.
Privacy redaction is best effort. Review labels and SQLite history do not provide enterprise
identity, tamper-proof audit storage or secure erasure.

## Rear-view examples and workspace update

The next pass replaces the dashboard's default example with three attributed real photographs of
workers seen from behind. Only exact hash-matched, visually reviewed samples may skip redaction;
uploaded images/videos and camera inputs keep their own privacy setting. The reason is retained
in each image run manifest. The original dataset images and pan video remain regression fixtures.

The interface now uses a light inspection layout, a compact header, a scene-first workflow,
collapsed advanced settings, quieter annotation labels and a dedicated observations/download panel.
Screenshots are captured from the working app. The new suite passes 190 tests on Python 3.11;
coverage is 81.26%. Real-model runs on the three photos returned 2, 4 and 4 detections respectively,
with 0, 0 and 1 events under their example policies. Those counts are functional observations,
not accuracy results. The browser smoke checks clear samples, ordinary upload redaction, a real
zone event, report hashes, H.264 playback and mobile layout.

Updates were pushed to `main` with the owner's authorization. [The complete CI matrix](https://github.com/MengyangGao/utility-site-safety-ai-system/actions/runs/34633162803)
passed at `ca01bec`: all nine OS/Python test jobs and the quality/browser/wheel job. A later documentation-only commit records these results without changing application source. No release, deployment or physical-camera operation was performed.


## Prediction review iteration (2.3.0)

Raw predictions now have an independent review workflow even when no policy event exists.
This addresses observations such as a CCTV fixture classified as gloves: the mistake can be
recorded without inventing a safety event. Review decisions are appended outside the run;
identifiers include the original log hash and line number. Reads and saves verify that hash,
and a changed run cannot inherit decisions from its previous evidence. Exports retain the
complete history with hashes. Detection and event review decisions remain separate.

Local validation: 198 tests passed on Python 3.11 (81.53% branch-aware coverage); Ruff and
mypy passed. AppTest saved a false-positive decision in a run with zero events, updated the
review counters, and verified original evidence bytes. The real Chromium/model workflow
passed image inference, prediction review display, ZIP evidence hashes, upload privacy,
zone events, H.264 playback and mobile layout. This iteration does not retrain the model
or claim improved detection accuracy.
