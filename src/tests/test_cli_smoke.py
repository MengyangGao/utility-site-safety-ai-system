"""Smoke tests for the CLI entry point."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType

from click.testing import CliRunner

from utility_safety_ai.cli import main


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Utility Site Safety AI" in result.output


def test_infer_image_help():
    runner = CliRunner()
    result = runner.invoke(main, ["infer-image", "--help"])
    assert result.exit_code == 0
    assert "--source" in result.output


def test_export_report(tmp_path):
    events_file = tmp_path / "events.jsonl"
    events_file.write_text('{"event_id":"e1","risk_level":"high","event_type":"zone_intrusion"}\n')
    output_dir = tmp_path / "reports"

    runner = CliRunner()
    result = runner.invoke(main, ["export-report", "--events", str(events_file), "--output", str(output_dir)])
    assert result.exit_code == 0
    assert (output_dir / "events.csv").exists()


def test_cli_exposes_user_commands_without_internal_quality_tools():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "fetch-model" in result.output
    assert "doctor" in result.output
    assert "export-model" in result.output
    assert "web" in result.output
    assert "model-gate" not in result.output
    assert "audit-provenance" not in result.output


def test_provenance_audit_command_passes_for_repository():
    repo_root = Path(__file__).resolve().parents[2]
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "audit-provenance",
            "--manifest",
            str(repo_root / "docs" / "legal" / "provenance.yaml"),
            "--repo-root",
            str(repo_root),
        ],
    )

    assert result.exit_code == 0, result.output
    assert '"passed": true' in result.output


def test_doctor_reports_environment_without_loading_model():
    runner = CliRunner()
    result = runner.invoke(main, ["doctor", "--model", "yolo11n.pt"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["model_state"] in {"local", "runtime-download"}
    assert payload["opencv"]


def test_cli_rejects_out_of_range_inference_parameters(tmp_path):
    source = tmp_path / "image.jpg"
    source.write_bytes(b"placeholder")
    runner = CliRunner()

    result = runner.invoke(
        main,
        ["infer-image", "--source", str(source), "--conf", "1.5"],
    )
    assert result.exit_code == 2
    assert "not in the range" in result.output


def test_cli_rejects_explicit_missing_zone_file(tmp_path):
    source = tmp_path / "image.jpg"
    source.write_bytes(b"placeholder")
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "infer-image",
            "--source",
            str(source),
            "--zones",
            str(tmp_path / "missing.yaml"),
        ],
    )
    assert result.exit_code == 2
    assert "does not exist" in result.output


def test_cli_reports_malformed_zone_config_before_loading_model(tmp_path):
    source = tmp_path / "image.jpg"
    source.write_bytes(b"placeholder")
    zones = tmp_path / "zones.yaml"
    zones.write_text("zones: not-a-list\n")
    runner = CliRunner()

    result = runner.invoke(
        main,
        ["infer-image", "--source", str(source), "--zones", str(zones)],
    )
    assert result.exit_code == 1
    assert "must be a list" in result.output


def test_fetch_model_moves_new_runtime_download_into_models(monkeypatch):
    fake_ultralytics = ModuleType("ultralytics")
    fake_ultralytics.__version__ = "test"

    class FakeYOLO:
        def __init__(self, model: str):
            Path(model).write_bytes(b"checkpoint")
            self.ckpt_path = model

    fake_ultralytics.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)
    runner = CliRunner()

    with runner.isolated_filesystem():
        result = runner.invoke(
            main,
            ["fetch-model", "--model", "yolo11n.pt", "--output", "models"],
        )

        assert result.exit_code == 0, result.output
        assert not Path("yolo11n.pt").exists()
        assert Path("models/yolo11n.pt").read_bytes() == b"checkpoint"
        assert Path("models/yolo11n.pt.json").is_file()


def test_fetch_model_force_preserves_old_checkpoint_when_copy_fails(monkeypatch):
    fake_ultralytics = ModuleType("ultralytics")
    fake_ultralytics.__version__ = "test"

    class FakeYOLO:
        def __init__(self, model: str):
            Path(model).write_bytes(b"new-checkpoint")
            self.ckpt_path = model

    fake_ultralytics.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)

    def fail_copy(*_args, **_kwargs):
        raise OSError("synthetic copy failure")

    monkeypatch.setattr("utility_safety_ai.cli.shutil.copy2", fail_copy)
    runner = CliRunner()

    with runner.isolated_filesystem():
        destination = Path("models/yolo11n.pt")
        destination.parent.mkdir()
        destination.write_bytes(b"old-checkpoint")

        result = runner.invoke(
            main,
            [
                "fetch-model",
                "--model",
                "yolo11n.pt",
                "--output",
                "models",
                "--force",
            ],
        )

        assert result.exit_code == 1
        assert destination.read_bytes() == b"old-checkpoint"
