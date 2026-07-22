"""Offline end-to-end tests for run-scoped image and video artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from utility_safety_ai.events.event import Detection
from utility_safety_ai.pipelines._artifacts import redact_source
from utility_safety_ai.pipelines.camera_pipeline import run_camera_pipeline
from utility_safety_ai.pipelines.image_pipeline import run_image_pipeline
from utility_safety_ai.pipelines.video_pipeline import run_video_pipeline
from utility_safety_ai.utils import paths as paths_module
from utility_safety_ai.utils.paths import resolve_latest_run


class FakeDetector:
    """Deterministic detector that exercises pipeline code without model I/O."""

    conf = 0.25
    iou = 0.45
    device = "cpu"
    model_path = "fake-ppe.pt"

    def __init__(self, detections: list[Detection] | None = None) -> None:
        self.detections = detections if detections is not None else _unsafe_person()
        self.reset_count = 0
        self.model = SimpleNamespace(
            ckpt_path=None,
            names={0: "person", 1: "no_helmet"},
        )

    def reset_tracking(self) -> None:
        self.reset_count += 1

    def predict(self, image: np.ndarray) -> list[Detection]:
        del image
        return list(self.detections)

    def track(self, image: np.ndarray) -> list[Detection]:
        return self.predict(image)


class FailingDetector(FakeDetector):
    def predict(self, image: np.ndarray) -> list[Detection]:
        del image
        raise RuntimeError("synthetic detector failure")


def _unsafe_person() -> list[Detection]:
    return [
        Detection(0, "person", 0.95, (12, 5, 52, 58)),
        Detection(1, "no_helmet", 0.90, (22, 8, 42, 25)),
    ]


def _write_image(path: Path) -> None:
    image = np.full((64, 80, 3), 180, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def _write_video(path: Path, frames: int = 3) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (80, 64),
    )
    assert writer.isOpened()
    for index in range(frames):
        frame = np.full((64, 80, 3), 80 + index * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    assert path.stat().st_size > 0


def _assert_audit_contract(run_dir: Path) -> dict:
    expected = [
        "events/events.jsonl",
        "events/events.csv",
        "events/detections.jsonl",
        "events/detections.csv",
        "events/compliance.jsonl",
        "events/compliance.csv",
        "events/summary.json",
        "events/summary.csv",
        "manifest.json",
    ]
    for relative in expected:
        assert (run_dir / relative).is_file(), relative
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["schema_version"] == "1.0"
    assert manifest["status"] == "completed"
    assert manifest["artifacts"]
    assert all(len(artifact["sha256"]) == 64 for artifact in manifest["artifacts"])
    assert manifest["runtime"]["utility_safety_ai_version"]
    assert manifest["runtime"]["python"]["version"]
    assert set(manifest["runtime"]["dependencies"]) == {
        "opencv",
        "torch",
        "ultralytics",
    }
    assert set(manifest["code"]) == {
        "commit_sha",
        "dirty",
        "unavailable_reason",
    }
    if manifest["code"]["commit_sha"] is not None:
        assert len(manifest["code"]["commit_sha"]) == 40
        assert isinstance(manifest["code"]["dirty"], bool)
    return manifest


def test_image_pipeline_creates_complete_run_without_overwriting_history(tmp_path):
    source = tmp_path / "input.jpg"
    _write_image(source)
    output = tmp_path / "outputs"

    detector = FakeDetector()
    annotated, events = run_image_pipeline(
        source,
        output,
        detector,
        zones=[],
        run_id="image-one",
        audit_source="worker-upload.jpg",
    )
    first_run = output / "runs" / "image-one"

    assert annotated.shape == (64, 80, 3)
    assert detector.reset_count == 1
    assert [event.event_type for event in events] == ["missing_helmet"]
    assert events[0].snapshot_path is not None
    assert not Path(events[0].snapshot_path).is_absolute()
    assert (first_run / events[0].snapshot_path).is_file()
    assert (first_run / "images" / "input_annotated.jpg").stat().st_size > 0
    assert len(list((first_run / "snapshots").glob("*.jpg"))) == 1
    manifest = _assert_audit_contract(first_run)
    assert manifest["source"]["value"] == "worker-upload.jpg"
    assert manifest["source"]["size_bytes"] == source.stat().st_size
    assert manifest["source"]["source_sha256"] == hashlib.sha256(
        source.read_bytes()
    ).hexdigest()
    assert manifest["model"]["model_kind"] == "ppe"
    assert manifest["model"]["capabilities"] == [
        "person_detection",
        "zone_intrusion",
        "ppe_detection",
        "ppe_rule_events",
    ]
    assert manifest["metrics"] == {"detections": 2, "events": 1, "frames_processed": 1}
    assert resolve_latest_run(output) == first_run.resolve()

    with pytest.raises(FileExistsError, match="already exists"):
        run_image_pipeline(source, output, FakeDetector([]), zones=[], run_id="image-one")
    assert resolve_latest_run(output) == first_run.resolve()

    run_image_pipeline(source, output, FakeDetector([]), zones=[], run_id="image-two")
    second_run = output / "runs" / "image-two"
    _assert_audit_contract(second_run)
    assert first_run.is_dir()
    assert resolve_latest_run(output) == second_run.resolve()

    # Empty JSONL logs and header-only CSV logs are intentional, valid artifacts.
    assert (second_run / "events" / "events.jsonl").read_text() == ""
    assert (second_run / "events" / "detections.jsonl").read_text() == ""
    assert (second_run / "events" / "compliance.jsonl").read_text() == ""
    assert (second_run / "events" / "events.csv").read_text().startswith("schema_version,")
    assert (second_run / "events" / "detections.csv").read_text().startswith("schema_version,")
    assert (second_run / "events" / "compliance.csv").read_text().startswith("person_track_id,")


def test_video_pipeline_processes_synthetic_video_and_persists_all_detections(tmp_path):
    source = tmp_path / "input.avi"
    _write_video(source)
    output = tmp_path / "outputs"
    detector = FakeDetector()

    events = run_video_pipeline(
        source,
        output,
        detector,
        zones=[],
        max_frames=3,
        run_id="video-e2e",
    )
    run_dir = output / "runs" / "video-e2e"
    manifest = _assert_audit_contract(run_dir)

    assert detector.reset_count == 1
    assert len(events) == 1  # Active every frame, logged once within cooldown.
    assert manifest["metrics"]["frames_processed"] == 3
    assert manifest["metrics"]["detections"] == 6
    assert manifest["source"]["value"] == source.name
    assert manifest["source"]["source_sha256"] == hashlib.sha256(
        source.read_bytes()
    ).hexdigest()
    assert (run_dir / "videos" / "input_annotated.mp4").stat().st_size > 0
    detection_lines = (run_dir / "events" / "detections.jsonl").read_text().splitlines()
    assert len(detection_lines) == 6
    assert {json.loads(line)["frame_index"] for line in detection_lines} == {0, 1, 2}


def test_video_pipeline_without_limit_processes_complete_video(tmp_path):
    source = tmp_path / "complete.avi"
    _write_video(source, frames=7)

    progress: list[tuple[int, int | None]] = []
    run_video_pipeline(
        source,
        tmp_path / "outputs",
        FakeDetector([]),
        zones=[],
        run_id="complete-video",
        progress_callback=lambda processed, total: progress.append((processed, total)),
    )

    manifest = _assert_audit_contract(
        tmp_path / "outputs" / "runs" / "complete-video"
    )
    assert manifest["config"]["max_frames"] is None
    assert manifest["metrics"]["frames_processed"] == 7
    assert progress[-1] == (7, 7)


def test_camera_pipeline_uses_monotonic_elapsed_time_and_checked_output(tmp_path):
    source = tmp_path / "camera-source.avi"
    _write_video(source)
    output = tmp_path / "outputs"
    detector = FakeDetector()

    events = run_camera_pipeline(
        str(source),
        output,
        detector,
        zones=[],
        max_frames=2,
        run_id="camera-e2e",
        audit_source="camera-7",
    )
    run_dir = output / "runs" / "camera-e2e"
    manifest = _assert_audit_contract(run_dir)

    assert detector.reset_count == 1
    assert len(events) == 1
    assert manifest["source"]["type"] == "camera"
    assert manifest["source"]["value"] == "camera-7"
    assert manifest["source"]["source_sha256"] is None
    assert manifest["source"]["size_bytes"] is None
    assert "no stable whole-input file" in manifest["source"][
        "integrity_unavailable_reason"
    ]
    assert manifest["metrics"]["frames_processed"] == 2
    assert (run_dir / "videos" / "camera_annotated.mp4").stat().st_size > 0
    rows = [
        json.loads(line)
        for line in (run_dir / "events" / "detections.jsonl").read_text().splitlines()
    ]
    elapsed = [row["time_seconds"] for row in rows if row["class_name"] == "person"]
    assert len(elapsed) == 2
    assert 0 <= elapsed[0] <= elapsed[1]


def test_invalid_input_creates_no_output_and_failed_run_does_not_replace_latest(tmp_path):
    output = tmp_path / "outputs"
    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_text("not an image")

    with pytest.raises(ValueError, match="decode"):
        run_image_pipeline(corrupt, output, FakeDetector(), zones=[])
    assert not output.exists()

    source = tmp_path / "valid.jpg"
    _write_image(source)
    run_image_pipeline(source, output, FakeDetector([]), zones=[], run_id="good")
    with pytest.raises(RuntimeError, match="synthetic detector failure"):
        run_image_pipeline(source, output, FailingDetector(), zones=[], run_id="failed")

    assert resolve_latest_run(output) == (output / "runs" / "good").resolve()
    failed_manifest = json.loads(
        (output / "runs" / "failed" / "manifest.json").read_text()
    )
    assert failed_manifest["status"] == "failed"
    assert failed_manifest["error"]["type"] == "RuntimeError"


def test_overwrite_is_transactional_across_failure_and_success(tmp_path):
    source = tmp_path / "input.jpg"
    _write_image(source)
    output = tmp_path / "outputs"
    run_dir = output / "runs" / "stable"

    run_image_pipeline(source, output, FakeDetector(), zones=[], run_id="stable")
    old_manifest = (run_dir / "manifest.json").read_bytes()
    old_image = (run_dir / "images" / "input_annotated.jpg").read_bytes()
    old_latest = (output / "latest.json").read_bytes()

    with pytest.raises(RuntimeError, match="synthetic detector failure"):
        run_image_pipeline(
            source,
            output,
            FailingDetector(),
            zones=[],
            run_id="stable",
            overwrite=True,
        )

    # A failed replacement never touches the published run or pointer.
    assert (run_dir / "manifest.json").read_bytes() == old_manifest
    assert (run_dir / "images" / "input_annotated.jpg").read_bytes() == old_image
    assert (output / "latest.json").read_bytes() == old_latest
    assert resolve_latest_run(output) == run_dir.resolve()
    diagnostics = list((output / "failed-runs").iterdir())
    assert len(diagnostics) == 1
    failed_manifest = json.loads((diagnostics[0] / "manifest.json").read_text())
    assert failed_manifest["status"] == "failed"
    assert failed_manifest["run_id"] == "stable"
    assert resolve_latest_run(output, include_failed=True) == diagnostics[0]

    # A completed replacement is swapped in only after all artifacts are ready.
    run_image_pipeline(
        source,
        output,
        FakeDetector([]),
        zones=[],
        run_id="stable",
        overwrite=True,
    )
    replacement = _assert_audit_contract(run_dir)
    assert replacement["metrics"]["detections"] == 0
    assert (run_dir / "manifest.json").read_bytes() != old_manifest
    assert not list((run_dir / "snapshots").iterdir())
    assert resolve_latest_run(output) == run_dir.resolve()
    assert not list((output / ".transactions").iterdir())


def test_latest_pointer_rejects_failed_manifest(tmp_path):
    source = tmp_path / "input.jpg"
    _write_image(source)
    output = tmp_path / "outputs"
    with pytest.raises(RuntimeError):
        run_image_pipeline(source, output, FailingDetector(), zones=[], run_id="failed")

    (output / "latest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "run_id": "failed",
                "run_dir": "runs/failed",
            }
        )
    )
    with pytest.raises(ValueError, match="completed run"):
        resolve_latest_run(output)


def test_overwrite_rolls_back_when_latest_publication_fails(tmp_path, monkeypatch):
    source = tmp_path / "input.jpg"
    _write_image(source)
    output = tmp_path / "outputs"
    run_dir = output / "runs" / "stable"
    run_image_pipeline(source, output, FakeDetector(), zones=[], run_id="stable")
    old_manifest = (run_dir / "manifest.json").read_bytes()
    old_latest = (output / "latest.json").read_bytes()
    real_atomic_write_json = paths_module._atomic_write_json

    def fail_latest(path: Path, value: dict) -> None:
        if path == output / "latest.json":
            raise OSError("synthetic latest publication failure")
        real_atomic_write_json(path, value)

    monkeypatch.setattr(paths_module, "_atomic_write_json", fail_latest)
    with pytest.raises(OSError, match="latest publication"):
        run_image_pipeline(
            source,
            output,
            FakeDetector([]),
            zones=[],
            run_id="stable",
            overwrite=True,
        )

    assert (run_dir / "manifest.json").read_bytes() == old_manifest
    assert (output / "latest.json").read_bytes() == old_latest
    assert resolve_latest_run(output) == run_dir.resolve()
    diagnostics = list((output / "failed-runs").iterdir())
    assert len(diagnostics) == 1
    assert json.loads((diagnostics[0] / "manifest.json").read_text())["status"] == "failed"


def test_source_redaction_removes_rtsp_credentials_and_tokens():
    redacted = redact_source(
        "rtsp://worker:password@camera.internal:8554/live?token=abc&quality=high#fragment"
    )
    assert redacted == "rtsp://camera.internal:8554/live?token=REDACTED&quality=high"
    assert "worker" not in redacted
    assert "password" not in redacted
    assert "abc" not in redacted
