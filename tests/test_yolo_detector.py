"""Offline tests for the Ultralytics detector adapter."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from utility_safety_ai.detection.yolo_detector import YoloDetector, _auto_device


class _Tensor:
    def __init__(self, value):
        self.value = value

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self.value)

    def item(self):
        return self.value


class _Boxes:
    def __init__(self):
        self.xyxy = [_Tensor([0, 0, 20, 40]), _Tensor([5, 0, 25, 40])]
        self.conf = [_Tensor(0.4), _Tensor(0.9)]
        self.cls = [_Tensor(0), _Tensor(0)]
        self.id = None

    def __len__(self):
        return 2


class _Result:
    boxes = _Boxes()
    names = {0: "person"}


class _FakeModel:
    def __init__(self):
        self.kwargs = None
        self.predictor = SimpleNamespace(trackers=[])

    def __call__(self, _image, **kwargs):
        self.kwargs = kwargs
        return [_Result()]


class _StreamingFakeModel(_FakeModel):
    def __call__(self, _image, **kwargs):
        self.kwargs = kwargs
        return iter([_Result()])


def _detector_without_ultralytics() -> YoloDetector:
    detector = YoloDetector.__new__(YoloDetector)
    detector.device = "cpu"
    detector.conf = 0.25
    detector.iou = 0.45
    detector.class_conf = {"person": 0.15, "gloves": 0.1}
    detector.inference_conf = 0.25
    detector.model = _FakeModel()
    return detector


def test_predict_submits_global_threshold_to_model():
    detector = _detector_without_ultralytics()
    detections = detector.predict(np.zeros((10, 10, 3), dtype=np.uint8))

    assert detector.model.kwargs["conf"] == 0.25
    assert len(detections) == 2


def test_predict_accepts_streaming_ultralytics_results():
    detector = _detector_without_ultralytics()
    detector.model = _StreamingFakeModel()

    detections = detector.predict(np.zeros((10, 10, 3), dtype=np.uint8))

    assert len(detections) == 2


def test_global_threshold_is_a_floor_for_safety_classes():
    detector = _detector_without_ultralytics()
    detector.conf = 0.8
    detector.inference_conf = 0.8

    detections = detector.predict(np.zeros((10, 10, 3), dtype=np.uint8))

    assert detector.model.kwargs["conf"] == 0.8
    assert [detection.confidence for detection in detections] == [0.9]


def test_adapter_does_not_apply_second_nms_to_nearby_people():
    detector = _detector_without_ultralytics()
    detections = detector._to_detections(_Result())
    assert len(detections) == 2


def test_auto_device_prefers_cuda_then_mps(monkeypatch):
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    assert _auto_device() == "cuda:0"

    fake_torch.cuda.is_available = lambda: False
    assert _auto_device() == "mps"


def test_reset_tracking_discards_predictor_state():
    detector = _detector_without_ultralytics()
    state = {"reset": 0}
    tracker = SimpleNamespace(reset=lambda: state.__setitem__("reset", 1))
    detector.model.predictor.trackers = [tracker]

    detector.reset_tracking()

    assert state["reset"] == 1
    assert detector.model.predictor is None
