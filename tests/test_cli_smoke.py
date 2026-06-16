"""Smoke tests for the CLI entry point."""

from __future__ import annotations

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
