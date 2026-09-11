"""Camera failure and backpressure tests, without physical camera access."""

import threading

import cv2
import numpy as np
import pytest

from utility_safety_ai.pipelines.capture import (
    CaptureError,
    CaptureOptions,
    FrameCapture,
    open_capture,
)

FRAME = np.zeros((32, 48, 3), dtype=np.uint8)


class BurstCapture:
    def __init__(self, count, stop, ready):
        self.count, self.stop, self.ready = count, stop, ready
        self.seen = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, _):
        return 25.0

    def read(self):
        if self.seen < self.count:
            self.seen += 1
            return True, FRAME.copy()
        self.ready.set()
        self.stop.wait(2)
        return False, None

    def release(self):
        self.released = True


def test_latest_frame_mailbox_discards_backlog():
    stop, ready = threading.Event(), threading.Event()
    source = BurstCapture(100, stop, ready)
    with FrameCapture(
        "rtsp://localhost/live", CaptureOptions(), stop_event=stop, opener=lambda *_: source
    ) as capture:
        assert ready.wait(2)
        packet = capture.read()
        assert packet.sequence == 100
        assert capture.frames_dropped == 99
        assert capture.summary()["buffer_policy"] == "latest_frame_only"
    assert source.released


def test_failed_stream_retries_are_bounded_and_redacted():
    calls = []

    class Failed:
        def isOpened(self):
            return False

        def release(self):
            calls.append("released")

    with FrameCapture(
        "rtsp://secret:password@localhost/live",
        CaptureOptions(reconnect_attempts=2, reconnect_delay_seconds=0),
        opener=lambda *_: Failed(),
    ) as capture:
        with pytest.raises(CaptureError) as error:
            capture.read()
        assert "password" not in str(error.value)
    assert len(calls) == 3


def test_reconnect_creates_a_new_continuity_segment():
    stop, disconnect, second_ready = threading.Event(), threading.Event(), threading.Event()
    first = BurstCapture(1, disconnect, threading.Event())
    second = BurstCapture(1, stop, second_ready)
    instances = iter([first, second])
    with FrameCapture(
        "rtsp://localhost/live",
        CaptureOptions(reconnect_delay_seconds=0),
        stop_event=stop,
        opener=lambda *_: next(instances),
    ) as capture:
        packet = capture.read()
        assert packet.segment == 0
        disconnect.set()
        assert second_ready.wait(2)
        packet = capture.read()
        assert packet.segment == 1
        assert capture.reconnections == 1
    assert first.released and second.released


def test_local_replay_preserves_every_frame(tmp_path):
    source_path = tmp_path / "replay.avi"
    source_path.touch()
    source = BurstCapture(3, threading.Event(), threading.Event())
    with FrameCapture(str(source_path), CaptureOptions(), opener=lambda *_: source) as capture:
        assert [capture.read().sequence for _ in range(3)] == [1, 2, 3]
        assert capture.frames_dropped == 0


def test_network_timeouts_are_passed_at_open(monkeypatch):
    calls = []
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: calls.append(args))
    open_capture(
        "rtsp://localhost/live", CaptureOptions(open_timeout_seconds=1.5, read_timeout_seconds=2)
    )
    assert calls[0][1] == cv2.CAP_FFMPEG
    assert calls[0][2] == [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
        1500,
        cv2.CAP_PROP_READ_TIMEOUT_MSEC,
        2000,
        cv2.CAP_PROP_N_THREADS,
        1,
    ]


def test_cancelled_capture_does_not_open_a_source():
    stop = threading.Event()
    stop.set()

    def unexpected(*_):
        raise AssertionError("Source should not open")

    with FrameCapture(
        "rtsp://localhost/live", CaptureOptions(), stop_event=stop, opener=unexpected
    ) as capture:
        assert capture.read() is None
    assert capture.stop_reason == "operator_stop"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"read_timeout_seconds": 0},
        {"open_timeout_seconds": float("nan")},
        {"reconnect_attempts": -1},
        {"reconnect_delay_seconds": float("inf")},
        {"max_frame_age_seconds": 0},
    ],
)
def test_capture_options_reject_invalid_limits(kwargs):
    with pytest.raises(ValueError):
        CaptureOptions(**kwargs)
