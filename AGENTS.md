AGENTS.md — Utility Site Safety AI System

> This file describes the implementation specification. The repository has been
> implemented as a v0.4 industrial-deliverable prototype with realistic CC0 construction/utility
> images, localized hazard zones, event aggregation, and optional custom PPE
> model training.

0. Mission

You are an autonomous AI coding agent. Your task is to implement an open-source project named:

Utility Site Safety AI System

This is a computer vision safety monitoring system for utility, construction, substation, electrical maintenance, and infrastructure worksite scenarios.

The project must support image and video inference, PPE compliance detection, restricted-zone intrusion detection, event logging, risk-level classification, privacy-preserving face blurring, a web demo, CLI tools, tests, documentation, and reproducible environment setup.

The final project should be suitable for:

1. A GitHub open-source portfolio project.
2. A Bilibili technical demo video.
3. A resume project for CLP Digital / Engineering / smart infrastructure / utility digitalisation roles.

Do not implement only a basic “helmet detection” demo. The system must be packaged as a practical engineering safety monitoring prototype.

⸻

1. Core Requirements

1.1 Environment

Use Miniconda to create a new Python environment.

Required environment name:

utility-safety-ai

Preferred Python version:

python=3.11

You may use Python 3.10 if any major dependency has compatibility issues, but document the reason clearly.

Create these files:

environment.yml
requirements.txt
README.md
AGENTS.md
Makefile

The project must be installable and runnable from a clean clone.

The setup should support:

conda env create -f environment.yml
conda activate utility-safety-ai
pip install -e .

or equivalent documented commands.

You are allowed to download pretrained models.

You are allowed to fine-tune models if useful.

You must not assume CUDA is available. The project must run on CPU. If CUDA, Apple Silicon MPS, or GPU acceleration is available, use it optionally, but the default path must remain CPU-compatible.

⸻

2. Project Scope

Implement a computer vision system with the following minimum capabilities:

2.1 PPE Detection

The system should detect at least:

person
helmet
vest
gloves
boots
goggles
no_helmet
no_vest
no_gloves
no_boots
no_goggles

If a pretrained PPE model with all classes is not available, implement a robust fallback:

1. Use a general YOLO model for person detection.
2. Allow loading a custom PPE model path.
3. Provide training / fine-tuning scripts for a PPE dataset.
4. Document how to train or fine-tune a PPE model.
5. Provide demo logic even if only partial classes are available.

Prefer using Ultralytics YOLO because it is practical and easy to reproduce.

⸻

2.2 Restricted-Zone Intrusion Detection

Implement restricted-zone detection.

The user must be able to define one or more polygonal zones in image/video coordinates.

When a detected person enters a restricted zone, the system should generate an event.

Support zone configuration via a JSON or YAML file, for example:

zones:
  - id: high_voltage_area_1
    name: High Voltage Area
    risk_level: high
    polygon:
      - [100, 200]
      - [500, 200]
      - [550, 600]
      - [80, 620]
  - id: crane_operation_area
    name: Crane Operation Area
    risk_level: medium
    polygon:
      - [600, 100]
      - [1100, 100]
      - [1150, 500]
      - [620, 520]

The system should check whether the bottom-center point of a person bounding box is inside the polygon.

⸻

2.3 Safety Rule Engine

Implement a rule engine that converts model detections and zone violations into safety events.

Minimum rules:

Missing helmet → medium risk
Missing vest → medium risk
Missing gloves → low or medium risk
Missing boots → medium risk
Missing goggles → medium risk
Person inside restricted zone → high risk
Missing helmet + inside restricted zone → critical risk
Multiple simultaneous PPE violations → high risk

The rule engine should be modular and easy to extend.

Implement event de-duplication:

1. Do not create a new event for every frame.
2. Merge repeated violations of the same type for the same tracked person within a configurable cooldown window.
3. Default cooldown: 10 seconds.

Tracking may be implemented with ByteTrack / BoT-SORT if available through Ultralytics, or with a simple IoU-based tracker as a fallback.

⸻

2.4 Event Logging

Every safety event must be saved.

Minimum event fields:

event_id
timestamp
source_type: image | video | camera
source_path
frame_index
time_seconds
risk_level
event_type
description
person_track_id
bbox
zone_id
zone_name
snapshot_path
metadata

Store events in:

outputs/events/events.jsonl
outputs/events/events.csv

Also persist every detection to:

outputs/events/detections.jsonl
outputs/events/detections.csv

For video inference, save annotated video to:

outputs/videos/

For images, save annotated images to:

outputs/images/

For event snapshots, save cropped or full-frame evidence images to:

outputs/snapshots/

⸻

2.5 Privacy Protection

Implement optional privacy-preserving processing.

Minimum requirement:

1. Add a CLI flag and web option:

--blur-faces

2. When enabled, blur detected faces before saving images/videos/snapshots.

If face detection is not available, implement a conservative fallback by blurring the upper portion of each person bounding box.

Document this clearly.

Do not store personally identifiable information.

Do not perform face recognition.

Do not identify individuals.

This project is for safety monitoring and engineering demonstration only.

⸻

2.6 Web Demo

Implement a web demo.

Preferred simple implementation:

Streamlit

The web app must support:

1. Upload image.
2. Upload video.
3. Select model path.
4. Configure confidence threshold.
5. Enable or disable face blurring.
6. Load or edit restricted-zone configuration.
7. Run inference.
8. Display annotated result.
9. Display safety events table.
10. Download CSV report.

Expected command:

streamlit run app.py

The app should be usable for a Bilibili demo video.

⸻

2.7 CLI

Implement a CLI command.

Suggested package command:

utility-safety-ai

Minimum CLI commands:

utility-safety-ai infer-image --source path/to/image.jpg --model path/to/model.pt --zones examples/zones.example.yaml --output outputs/
utility-safety-ai infer-video --source path/to/video.mp4 --model path/to/model.pt --zones examples/zones.example.yaml --output outputs/
utility-safety-ai train --data path/to/data.yaml --model yolo11n.pt --epochs 30
utility-safety-ai export-report --events outputs/events/events.jsonl --format csv

If using a simpler CLI structure, document it clearly.

⸻

3. Recommended Repository Structure

Create a clean, maintainable structure:

utility-safety-ai/
  AGENTS.md
  README.md
  LICENSE
  Makefile
  environment.yml
  requirements.txt
  pyproject.toml
  app.py
  examples/
    zones.example.yaml
    rules.example.yaml
    model.example.yaml
  src/
    utility_safety_ai/
      __init__.py
      cli.py
      detection/
        yolo_detector.py
        model_loader.py
      tracking/
        simple_tracker.py
      zones/
        zone.py
        zone_loader.py
        geometry.py
      rules/
        rule_engine.py
        risk.py
      privacy/
        face_blur.py
      events/
        event.py
        event_logger.py
        detection_logger.py
        summary.py
        report.py
      pipelines/
        image_pipeline.py
        video_pipeline.py
      visualization/
        annotator.py
      training/
        train_yolo.py
      utils/
        paths.py
        video.py
        logging.py
  tests/
    test_geometry.py
    test_zone_loader.py
    test_rule_engine.py
    test_event_logger.py
    test_privacy_blur.py
    test_cli_smoke.py
  examples/
    sample_zones.yaml
    sample_images/
    sample_videos/
  outputs/
    .gitkeep

You may adjust the structure if necessary, but keep it clean and documented.

⸻

4. Model and Dataset Requirements

4.1 Pretrained Models

Download a lightweight YOLO model automatically if no model is provided.

Recommended default:

yolo11n.pt or yolov8n.pt

Use the latest stable Ultralytics-supported model available in the environment.

If a model cannot detect PPE classes, the system must still run and provide person + restricted zone detection.

Default model auto-discovery order:

1. `models/ppe_yolo11n.pt` if present — fast CPU-friendly PPE model (default).
2. `models/ppe_yolo11s.pt` if present — slightly stronger, slower PPE alternative.
3. `yolo11n.pt` otherwise — Ultralytics downloads it automatically.

Document clearly:

1. Default model behavior.
2. How to use a custom PPE model.
3. How to fine-tune on a PPE dataset.
4. Expected dataset format.

⸻

4.2 Training / Fine-tuning

Implement a training script using Ultralytics.

Expected usage:

python -m utility_safety_ai.training.train_yolo \
  --data datasets/construction_ppe/data.yaml \
  --model yolo11n.pt \
  --epochs 30 \
  --imgsz 640

or:

utility-safety-ai train --data datasets/construction_ppe/data.yaml --model yolo11n.pt --epochs 30

Training should not be required for the basic demo to run.

If training cannot be completed locally because dataset is missing, provide clear instructions and tests that do not depend on the dataset.

⸻

5. Software Quality Requirements

5.1 Tests Are Mandatory

You must implement tests before final reporting.

Use:

pytest

Minimum tests:

1. Point-in-polygon geometry test.
2. Zone YAML loading test.
3. Rule engine test for PPE violation.
4. Rule engine test for restricted-zone violation.
5. Critical risk escalation test.
6. Event logger JSONL/CSV writing test.
7. Privacy blur smoke test.
8. CLI smoke test.

All tests must pass before you report completion.

Do not report success until the test suite passes.

Required verification command:

pytest -q

Also run basic lint or formatting if configured.

Recommended:

ruff check .

If ruff is installed, fix all reasonable issues.

⸻

5.2 Acceptance Criteria

The project is complete only if all of these are true:

1. A fresh conda environment can be created.
2. The package can be installed.
3. Image inference runs on at least one sample image.
4. Video inference runs on at least one short sample video or generated synthetic video.
5. The web app starts without import errors.
6. Events are saved to JSONL and CSV.
7. Annotated outputs are saved.
8. Face/person privacy blur option works.
9. Restricted-zone intrusion detection works.
10. Tests pass with pytest -q.
11. README includes setup, usage, project motivation, architecture, limitations, and resume bullet examples.

⸻

6. README Requirements

The README must be written professionally.

It should include:

6.1 Project Positioning

Explain that this is an open-source engineering prototype for utility and construction safety monitoring.

Mention use cases:

substation maintenance
utility construction sites
electrical cabinet work
EV charging facility construction
infrastructure worksite safety

Do not claim it is production-ready.

Use responsible wording:

This project is an educational and engineering demonstration prototype. It is not certified for real safety-critical deployment.

⸻

6.2 Features

List:

PPE compliance detection
Restricted-zone intrusion detection
Risk-level rule engine
Event logging
Annotated image/video output
Privacy-preserving face/person blur
Web demo
CLI tools
YOLO fine-tuning support

⸻

6.3 Architecture

Include an ASCII architecture diagram, for example:

Image/Video/Camera
      ↓
YOLO Detector
      ↓
Tracker
      ↓
Zone Intrusion Checker
      ↓
Safety Rule Engine
      ↓
Event Logger + Annotator
      ↓
Dashboard / Reports / Annotated Outputs

⸻

6.4 Setup

Include exact commands:

conda env create -f environment.yml
conda activate utility-safety-ai
pip install -e .

⸻

6.5 Usage

Include image, video, web app, and training examples.

⸻

6.6 Resume Bullets

Include resume-ready bullet points.

Example:

Built an open-source computer vision safety monitoring system for utility and construction scenarios, supporting PPE compliance detection, restricted-zone intrusion alerts, event logging, privacy-preserving face blurring, and web-based inspection reports.
Designed a modular safety rule engine that converts YOLO detections and restricted-zone geometry into risk-ranked safety events, with event de-duplication and CSV/JSONL reporting.

⸻

6.7 Limitations

Be honest.

Mention:

Model performance depends on dataset quality.
Default YOLO models may not detect all PPE classes without fine-tuning.
The system is not certified for safety-critical use.
Restricted-zone accuracy depends on camera calibration and zone configuration.
Face blurring is a privacy aid, not a full anonymisation guarantee.

⸻

7. Implementation Details

7.1 Geometry

Implement reliable polygon checking.

Function example:

def point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    ...

Do not rely on heavy GIS libraries unless necessary.

⸻

7.2 Detection Output Format

Normalize detection outputs into a common dataclass:

@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]
    track_id: int | None = None

⸻

7.3 Safety Event Format

Use dataclass or Pydantic model:

@dataclass
class SafetyEvent:
    event_id: str
    timestamp: str
    source_type: str
    source_path: str
    frame_index: int | None
    time_seconds: float | None
    risk_level: str
    event_type: str
    description: str
    person_track_id: int | None
    bbox: tuple[float, float, float, float] | None
    zone_id: str | None
    zone_name: str | None
    snapshot_path: str | None
    metadata: dict

⸻

7.4 Risk Levels

Use fixed risk levels:

low
medium
high
critical

Implement deterministic escalation logic.

⸻

7.5 Generated Synthetic Test Data

If no sample video/image exists, generate synthetic test assets for tests.

For example:

1. Create a blank image.
2. Draw a fake person bounding box.
3. Define a restricted zone.
4. Test the geometry, rule engine, logger, and annotation independently from YOLO.

Tests should not require internet or model downloads.

Model download and real inference can be smoke-tested separately if network is available.

⸻

8. Engineering Constraints

1. Keep the code modular.
2. Avoid hard-coded absolute paths.
3. Use pathlib.
4. Use type hints.
5. Use dataclasses or Pydantic for structured data.
6. Handle missing model files gracefully.
7. Handle missing zone files gracefully.
8. Handle corrupted video files gracefully.
9. Avoid saving huge outputs by default.
10. Keep default models small enough for a laptop.

⸻

9. Safety and Ethics

This project must avoid unsafe or irresponsible claims.

Do not claim:

This guarantees worker safety.
This replaces human safety officers.
This is production-certified.

Do state:

This is a prototype decision-support and demonstration system.
Human review is required.
The system may generate false positives and false negatives.
Privacy protection should be enabled when processing real people.

Do not implement face recognition.

Do not implement identity tracking beyond temporary per-video track IDs.

⸻

10. Final Report Requirement

Before reporting completion, run:

pytest -q

Also run at least one demo command, for example:

utility-safety-ai infer-image \
  --source examples/sample_images/demo.jpg \
  --zones examples/zones.example.yaml \
  --output outputs/demo

If possible, also run:

streamlit run app.py

or at least verify:

python -m py_compile app.py

The final report must include:

1. What was implemented.
2. Exact environment setup commands.
3. Exact test command and result.
4. Exact demo command used.
5. Output file locations.
6. Known limitations.
7. Next recommended improvements.

Do not report “done” unless the tests pass and the demo command runs successfully.

⸻

11. Suggested Implementation Plan

Follow this order:

Phase 1 — Skeleton

Create project structure, pyproject, environment files, README draft, CLI skeleton.

Phase 2 — Core Logic

Implement:

Detection dataclass
SafetyEvent dataclass
Zone loader
Point-in-polygon
Rule engine
Event logger
Annotator
Privacy blur

Write tests for these modules.

Phase 3 — Inference Pipeline

Implement YOLO model loading, image pipeline, video pipeline, output saving.

Phase 4 — Web App

Implement Streamlit app with image/video upload, zone config, model selection, result display, and report download.

Phase 5 — Training Support

Implement YOLO fine-tuning script and documentation.

Phase 6 — Tests and Acceptance

Run all tests, fix errors, run demo, update README, then report.

⸻

12. Definition of Done

The project is considered done only when:

conda activate utility-safety-ai
pytest -q

passes, and at least one image or video demo produces:

annotated output
events.jsonl
events.csv
snapshot images

Do not skip tests.

Do not skip documentation.

Do not skip acceptance verification.
