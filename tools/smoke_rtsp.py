"""Real loopback RTSP fault test: FFmpeg -> MediaMTX -> the bundled PPE detector.

Requires mediamtx and ffmpeg on PATH. Starts only temporary, loopback-bound services.
No physical camera or external worksite stream is accessed.
"""

import argparse
import json
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from utility_safety_ai.detection.yolo_detector import YoloDetector
from utility_safety_ai.pipelines.camera_pipeline import run_camera_pipeline
from utility_safety_ai.pipelines.capture import CaptureOptions
from utility_safety_ai.resources import resource_root
from utility_safety_ai.zones.zone_loader import load_zones


def unused_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def stop_process(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/rtsp-smoke"))
    parser.add_argument(
        "--steady",
        action="store_true",
        help="Require 30 frames without reconnecting; do not interrupt the publisher.",
    )
    parser.add_argument(
        "--capture-only", action="store_true", help="Probe transport without model inference."
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for tool in ("mediamtx", "ffmpeg"):
        if not shutil.which(tool):
            raise SystemExit(f"Install {tool} before running this optional integration check.")
    root = resource_root()
    if args.capture_only:
        detector = type(
            "TransportProbe",
            (),
            {"track": lambda self, frame: [], "reset_tracking": lambda self: None},
        )()
    else:
        detector = YoloDetector(root / "models/ppe_yolo11n.pt", device="cpu")
    zones = load_zones(root / "examples/zones_construction_site_ppe_01.yaml")
    port = unused_port()
    uri = f"rtsp://127.0.0.1:{port}/smoke"
    from utility_safety_ai.utils.paths import generate_run_id

    run_id = generate_run_id()
    run_dir = args.output / "runs" / run_id
    processes = []
    stop = threading.Event()
    restart_thread = None
    with tempfile.TemporaryDirectory() as tmp:
        config = Path(tmp) / "mediamtx.yml"
        config.write_text(
            f"logLevel: warn\nrtspAddress: 127.0.0.1:{port}\nrtspTransports: [tcp]\nrtmp: false\nhls: false\nwebrtc: false\nsrt: false\npaths:\n  smoke:\n"
        )
        with (args.output / "services.log").open("w") as log:
            server = subprocess.Popen(["mediamtx", str(config)], cwd=tmp, stdout=log, stderr=log)
            processes.append(server)
            try:
                for _ in range(100):
                    try:
                        with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                            break
                    except OSError:
                        if server.poll() is not None:
                            raise RuntimeError(
                                "MediaMTX exited; check its local configuration."
                            ) from None
                        time.sleep(0.05)

                def publish():
                    process = subprocess.Popen(
                        [
                            "ffmpeg",
                            "-nostdin",
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-re",
                            "-stream_loop",
                            "-1",
                            "-i",
                            str(root / "examples/sample_videos/construction_ppe_pan.mp4"),
                            "-an",
                            "-c:v",
                            "libx264",
                            "-preset",
                            "ultrafast",
                            "-tune",
                            "zerolatency",
                            "-pix_fmt",
                            "yuv420p",
                            "-g",
                            "15",
                            "-f",
                            "rtsp",
                            "-rtsp_transport",
                            "tcp",
                            uri,
                        ],
                        stdout=log,
                        stderr=log,
                    )
                    processes.append(process)
                    return process

                publisher = publish()
                time.sleep(0.5)
                did_interrupt = False

                def observe(_frame, health):
                    nonlocal did_interrupt, restart_thread
                    if not args.steady and health["frames"] >= 5 and not did_interrupt:
                        did_interrupt = True
                        stop_process(publisher)

                        def restart():
                            if not stop.wait(0.8):
                                publish()

                        restart_thread = threading.Thread(target=restart)
                        restart_thread.start()
                    if (args.steady and health["frames"] >= 30) or (
                        not args.steady and health["stream_segment"] >= 1 and health["frames"] >= 12
                    ):
                        stop.set()

                run_camera_pipeline(
                    uri,
                    args.output,
                    detector,
                    zones,
                    duration_seconds=30,
                    blur_faces_enabled=True,
                    run_id=run_id,
                    capture_options=CaptureOptions(
                        open_timeout_seconds=5,
                        read_timeout_seconds=5,
                        reconnect_attempts=6,
                        reconnect_delay_seconds=0.2,
                    ),
                    stop_event=stop,
                    frame_callback=observe,
                )
                manifest = json.loads((run_dir / "manifest.json").read_text())
                assert manifest["status"] == "completed"
                if args.steady:
                    assert manifest["metrics"]["reconnections"] == 0
                    assert manifest["metrics"]["frames_processed"] >= 30
                else:
                    assert manifest["metrics"]["reconnections"] >= 1
                rows = [
                    json.loads(line)
                    for line in (run_dir / "events/frame_timestamps.jsonl").read_text().splitlines()
                ]
                if not args.steady:
                    assert max(row["stream_segment"] for row in rows) >= 1
                assert rows == sorted(rows, key=lambda row: row["capture_elapsed_seconds"])
                result = {
                    "scope": "Real loopback RTSP replay. No physical camera, latency benchmark or accuracy claim.",
                    "mode": "steady" if args.steady else "interrupted publisher",
                    "detector": "transport probe" if args.capture_only else "bundled PPE model",
                    "result": "passed",
                    "model_sha256": manifest["model"]["model_sha256"],
                    "metrics": manifest["metrics"],
                }
                (args.output / "integration-result.json").write_text(
                    json.dumps(result, indent=2) + "\n"
                )
                print(json.dumps(result, indent=2))
            finally:
                stop.set()
                if restart_thread:
                    restart_thread.join(timeout=3)
                for process in reversed(processes):
                    stop_process(process)


if __name__ == "__main__":
    main()
