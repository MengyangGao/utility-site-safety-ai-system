"""The model adapter opts out in this process without rewriting global user settings."""

import sys
from types import ModuleType
from unittest.mock import Mock

from utility_safety_ai.detection import runtime


def test_local_loader_sets_offline_before_import_and_disables_usage_events(monkeypatch):
    module = ModuleType("ultralytics")
    module.YOLO = Mock(return_value="model")
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    monkeypatch.delenv("YOLO_OFFLINE", raising=False)
    disable = Mock()
    monkeypatch.setattr(runtime, "_disable_usage_events", disable)
    assert runtime.load_yolo("local.pt") == "model"
    assert runtime.os.environ["YOLO_OFFLINE"] == "1"
    disable.assert_called_once()


def test_explicit_download_command_can_enable_networking(monkeypatch):
    module = ModuleType("ultralytics")
    module.YOLO = Mock()
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    monkeypatch.delenv("YOLO_OFFLINE", raising=False)
    monkeypatch.setattr(runtime, "_disable_usage_events", Mock())
    runtime.load_yolo("model.pt", allow_download=True)
    assert runtime.os.environ["YOLO_OFFLINE"] == "0"
    monkeypatch.setenv("YOLO_OFFLINE", "1")
    runtime.load_yolo("model.pt", allow_download=True)
    assert runtime.os.environ["YOLO_OFFLINE"] == "1"
