"""Bounded live capture, latest-frame backpressure and explicit stream continuity."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class CaptureError(RuntimeError):
    """A source failed without exposing its credential-bearing URI."""


@dataclass(frozen=True)
class CaptureOptions:
    open_timeout_seconds: float = 5.0
    read_timeout_seconds: float = 5.0
    reconnect_attempts: int = 3
    reconnect_delay_seconds: float = 0.5
    max_frame_age_seconds: float = 2.0

    def __post_init__(self) -> None:
        for name in ("open_timeout_seconds", "read_timeout_seconds", "max_frame_age_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and greater than zero")
        if (
            isinstance(self.reconnect_attempts, bool)
            or not isinstance(self.reconnect_attempts, int)
            or self.reconnect_attempts < 0
        ):
            raise ValueError("reconnect_attempts must be a non-negative integer")
        if not isfinite(self.reconnect_delay_seconds) or self.reconnect_delay_seconds < 0:
            raise ValueError("reconnect_delay_seconds must be finite and non-negative")


@dataclass(frozen=True)
class CapturedFrame:
    image: np.ndarray
    captured_at: float
    sequence: int
    segment: int
    fps: float


def is_network_source(source: str | int) -> bool:
    return isinstance(source, str) and source.lower().startswith(
        ("rtsp://", "rtsps://", "http://", "https://")
    )


def open_capture(source: str | int, options: CaptureOptions):
    if is_network_source(source):
        if str(source).lower().startswith(("rtsp://", "rtsps://")):
            # OpenCV exposes RTSP transport through a process-wide FFmpeg option.
            # Respect an operator override; choose TCP for the single-camera demo.
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        # These are open-only FFmpeg properties. Calling set() after construction
        # does not reliably establish a timeout and must not be used as a fallback.
        return cv2.VideoCapture(
            source,
            cv2.CAP_FFMPEG,
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                max(1, round(options.open_timeout_seconds * 1000)),
                cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                max(1, round(options.read_timeout_seconds * 1000)),
                cv2.CAP_PROP_N_THREADS,
                1,
            ],
        )
    return cv2.VideoCapture(source)


class FrameCapture:
    """A one-frame mailbox for live sources; local file replay remains sequential.

    One thread owns a live VideoCapture from open through release. A slow detector
    receives the most recent decoded frame instead of accumulating a stale queue.
    Frame age measures time in this process, not camera-to-host network latency.
    """

    def __init__(
        self,
        source: str | int,
        options: CaptureOptions,
        *,
        stop_event: threading.Event | None = None,
        deadline: float | None = None,
        opener=None,
    ):
        self.source = int(source) if isinstance(source, str) and source.isdigit() else source
        self.options = options
        self.stop = stop_event or threading.Event()
        self.deadline = deadline
        self.opener = opener or open_capture
        self.file_replay = (
            isinstance(self.source, str)
            and not is_network_source(self.source)
            and Path(self.source).is_file()
        )
        self._condition = threading.Condition()
        self._latest: CapturedFrame | None = None
        self._finished = False
        self._error: CaptureError | None = None
        self._thread: threading.Thread | None = None
        self._file_capture: Any = None
        self.frames_received = 0
        self.frames_dropped = 0
        self.stale_frames = 0
        self.read_failures = 0
        self.reconnections = 0
        self.stop_reason = "running"

    def __enter__(self):
        if self.file_replay:
            self._file_capture = self.opener(self.source, self.options)
            if not self._file_capture.isOpened():
                self._file_capture.release()
                raise CaptureError("Could not open the source file.")
        else:
            self._thread = threading.Thread(
                target=self._produce, name="safety-capture", daemon=True
            )
            self._thread.start()
        return self

    def _stopping(self) -> bool:
        if self.deadline is not None and time.monotonic() >= self.deadline:
            self.stop_reason = "duration_limit"
            return True
        if self.stop.is_set():
            if self.stop_reason == "running":
                self.stop_reason = "operator_stop"
            return True
        return False

    def _packet(self, capture, frame, segment: int) -> CapturedFrame:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        self.frames_received += 1
        return CapturedFrame(
            frame,
            time.monotonic(),
            self.frames_received,
            segment,
            fps if 0.1 <= fps <= 240 else 30.0,
        )

    def _produce(self) -> None:
        failures = 0
        segment = 0
        connected_before = False
        try:
            while not self._stopping():
                capture = self.opener(self.source, self.options)
                received_in_segment = False
                try:
                    if capture.isOpened():
                        while not self._stopping():
                            ok, frame = capture.read()
                            if not ok or frame is None or not frame.size:
                                break
                            if not received_in_segment:
                                if connected_before:
                                    segment += 1
                                    self.reconnections += 1
                                received_in_segment = connected_before = True
                            failures = 0
                            packet = self._packet(capture, frame, segment)
                            with self._condition:
                                if self._latest is not None:
                                    self.frames_dropped += 1
                                self._latest = packet
                                self._condition.notify_all()
                finally:
                    capture.release()
                if self._stopping():
                    break
                self.read_failures += 1
                failures += 1
                with self._condition:
                    # Do not serve an old mailbox frame as current after a connection gap.
                    self._latest = None
                if not is_network_source(self.source) or failures > self.options.reconnect_attempts:
                    raise CaptureError("Capture unavailable after the configured reconnect budget.")
                delay = min(self.options.reconnect_delay_seconds * 2 ** min(failures - 1, 10), 8.0)
                if self.deadline is not None:
                    delay = min(delay, max(0.0, self.deadline - time.monotonic()))
                self.stop.wait(delay)
        except Exception:
            self._error = CaptureError("Capture unavailable after the configured reconnect budget.")
            self.stop_reason = "capture_failed"
        finally:
            with self._condition:
                self._finished = True
                self._condition.notify_all()

    def read(self) -> CapturedFrame | None:
        if self._stopping():
            return None
        if self.file_replay:
            ok, frame = self._file_capture.read()
            if not ok or frame is None or not frame.size:
                self.stop_reason = "end_of_file"
                return None
            return self._packet(self._file_capture, frame, 0)
        wait_deadline = time.monotonic() + (
            self.options.open_timeout_seconds + self.options.read_timeout_seconds + 8
        ) * (self.options.reconnect_attempts + 1)
        with self._condition:
            while not self._stopping():
                if time.monotonic() >= wait_deadline:
                    raise CaptureError("No frame arrived within the bounded capture wait.")
                if self._error:
                    raise self._error
                if self._latest is not None:
                    packet, self._latest = self._latest, None
                    if time.monotonic() - packet.captured_at <= self.options.max_frame_age_seconds:
                        return packet
                    self.stale_frames += 1
                if self._finished:
                    return None
                self._condition.wait(timeout=0.1)
        return None

    def summary(self) -> dict:
        return {
            "frames_received": self.frames_received,
            "frames_dropped": self.frames_dropped,
            "stale_frames_discarded": self.stale_frames,
            "read_failures": self.read_failures,
            "reconnections": self.reconnections,
            "stop_reason": self.stop_reason,
            "buffer_policy": "sequential" if self.file_replay else "latest_frame_only",
            "capture_options": asdict(self.options),
        }

    def __exit__(self, exc_type, exc, traceback):
        self.stop.set()
        if self._file_capture is not None:
            self._file_capture.release()
        if self._thread is not None:
            self._thread.join(
                self.options.open_timeout_seconds + self.options.read_timeout_seconds + 1
            )
            if self._thread.is_alive() and exc is None:
                raise CaptureError(
                    "Capture backend did not stop within its timeout; run this backend in a supervised process."
                )
