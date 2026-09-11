"""Tests for deterministic model path resolution."""

from __future__ import annotations

import pytest

from utility_safety_ai.detection import model_loader
from utility_safety_ai.detection.model_loader import DEFAULT_MODEL, resolve_model_path


def test_none_model_uses_downloadable_default_when_no_local_weights(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(model_loader, "PPE_MODEL_CANDIDATES", [])
    monkeypatch.setattr(model_loader, "DEFAULT_MODEL_LOCAL_CANDIDATES", [])
    assert resolve_model_path(None) == DEFAULT_MODEL


def test_none_model_reuses_fetched_local_general_model(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(model_loader, "PPE_MODEL_CANDIDATES", [])
    model = tmp_path / "models" / "yolo11n.pt"
    model.parent.mkdir()
    model.write_bytes(b"general weights")

    assert resolve_model_path(None) == model


def test_local_ppe_model_takes_priority_over_general_model(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    ppe_model = tmp_path / "models" / "ppe_yolo11n.pt"
    general_model = tmp_path / "models" / "yolo11n.pt"
    ppe_model.parent.mkdir()
    ppe_model.write_bytes(b"ppe weights")
    general_model.write_bytes(b"general weights")

    assert resolve_model_path(None) == ppe_model


def test_existing_explicit_model_path_is_returned(tmp_path):
    model = tmp_path / "custom.pt"
    model.write_bytes(b"weights")
    assert resolve_model_path(model) == model


def test_bare_ultralytics_name_is_allowed_for_auto_download(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert resolve_model_path("yolo11s.pt") == "yolo11s.pt"


def test_missing_explicit_model_path_fails_fast(tmp_path):
    missing = tmp_path / "models" / "missing.pt"
    with pytest.raises(FileNotFoundError, match="weights file not found"):
        resolve_model_path(missing)
