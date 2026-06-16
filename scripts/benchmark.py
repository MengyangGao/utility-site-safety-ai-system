"""Benchmark inference FPS across CPU, MPS, and CUDA if available."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _synthetic_frame(width: int = 640, height: int = 640) -> np.ndarray:
    return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)


def benchmark(
    model_path: str,
    device: str,
    warmup: int = 5,
    iterations: int = 50,
    image_size: int = 640,
) -> dict:
    """Run a simple FPS benchmark for the supplied model and device."""
    from utility_safety_ai.detection.yolo_detector import YoloDetector

    detector = YoloDetector(model_path=model_path, device=device, conf=0.25, iou=0.45)
    frame = _synthetic_frame(image_size, image_size)

    logger.info("Warming up %s on %s ...", model_path, device)
    for _ in range(warmup):
        detector.predict(frame)

    logger.info("Benchmarking %s iterations on %s ...", iterations, device)
    start = time.perf_counter()
    for _ in range(iterations):
        detector.predict(frame)
    elapsed = time.perf_counter() - start

    fps = iterations / elapsed
    ms_per_frame = (elapsed / iterations) * 1000
    return {
        "model": str(model_path),
        "device": device,
        "image_size": image_size,
        "iterations": iterations,
        "total_seconds": round(elapsed, 3),
        "fps": round(fps, 2),
        "ms_per_frame": round(ms_per_frame, 2),
    }


def available_devices() -> list[str]:
    devices = ["cpu"]
    try:
        import torch

        if torch.backends.mps.is_available():
            devices.append("mps")
        if torch.cuda.is_available():
            devices.append("cuda")
    except Exception:
        pass
    return devices


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PPE model inference FPS.")
    parser.add_argument("--model", default="models/ppe_yolo11n.pt", help="Model path or name.")
    parser.add_argument("--device", default=None, help="Specific device to benchmark.")
    parser.add_argument("--iterations", default=50, type=int, help="Number of inference iterations.")
    parser.add_argument("--output", default="outputs/benchmark", help="Output directory.")
    args = parser.parse_args()

    devices = [args.device] if args.device else available_devices()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for device in devices:
        try:
            result = benchmark(args.model, device, iterations=args.iterations)
            results.append(result)
            logger.info(
                "Device=%s | FPS=%.2f | ms/frame=%.2f",
                device,
                result["fps"],
                result["ms_per_frame"],
            )
        except Exception as exc:
            logger.error("Benchmark failed for device %s: %s", device, exc)
            results.append(
                {
                    "model": str(args.model),
                    "device": device,
                    "error": str(exc),
                }
            )

    out_path = output_dir / "benchmark.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Benchmark results saved to %s", out_path)


if __name__ == "__main__":
    main()
