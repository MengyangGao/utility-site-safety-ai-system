"""Offline behavior tests for the YOLO training wrapper."""

from __future__ import annotations

import pytest

from utility_safety_ai.training import train_yolo


class _FakeYOLO:
    checkpoint: dict | None = None
    last_instance: _FakeYOLO | None = None

    def __init__(self, model: str):
        self.model = model
        self.ckpt = self.checkpoint
        self.train_kwargs = None
        type(self).last_instance = self

    def train(self, **kwargs):
        self.train_kwargs = kwargs


def test_training_auto_device_is_forwarded(monkeypatch):
    monkeypatch.setattr(train_yolo, "load_yolo", lambda model, **kwargs: _FakeYOLO(model))
    monkeypatch.setattr(train_yolo, "_auto_device", lambda: "cuda:0")

    train_yolo.train("construction-ppe.yaml", epochs=1, device=None)

    assert _FakeYOLO.last_instance is not None
    assert _FakeYOLO.last_instance.train_kwargs["device"] == "cuda:0"


def test_resume_rejects_stripped_completed_checkpoint(tmp_path, monkeypatch):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"placeholder")
    _FakeYOLO.checkpoint = {"epoch": -1, "optimizer": None}
    monkeypatch.setattr(train_yolo, "load_yolo", lambda model, **kwargs: _FakeYOLO(model))

    with pytest.raises(ValueError, match="unstripped"):
        train_yolo.train("data.yaml", model=str(checkpoint), resume=True)


def test_resume_forwards_stateful_interrupted_checkpoint(tmp_path, monkeypatch):
    checkpoint = tmp_path / "last.pt"
    checkpoint.write_bytes(b"placeholder")
    _FakeYOLO.checkpoint = {"epoch": 4, "optimizer": {"state": {}}}
    monkeypatch.setattr(train_yolo, "load_yolo", lambda model, **kwargs: _FakeYOLO(model))

    train_yolo.train("data.yaml", model=str(checkpoint), resume=True)

    assert _FakeYOLO.last_instance is not None
    assert _FakeYOLO.last_instance.train_kwargs["resume"] is True


def test_resume_requires_existing_checkpoint(monkeypatch):
    monkeypatch.setattr(train_yolo, "load_yolo", lambda model, **kwargs: _FakeYOLO(model))

    with pytest.raises(FileNotFoundError, match="existing"):
        train_yolo.train("data.yaml", model="missing-last.pt", resume=True)
