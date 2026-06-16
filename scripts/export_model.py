"""Export a trained YOLO model to ONNX/TensorRT/TorchScript formats."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SUPPORTED_FORMATS = ["onnx", "torchscript", "openvino", "engine", "tflite", "ncnn", "coreml"]


def export(
    model_path: str,
    fmt: str,
    imgsz: int = 640,
    output_dir: str | Path = "outputs/export",
    half: bool = False,
    int8: bool = False,
    dynamic: bool = False,
    simplify: bool = False,
    data: str | None = None,
    workspace: int | None = None,
) -> Path:
    """Export a YOLO model to the requested deployment format.

    Optimization options:
      * ``half``  → FP16 weights (supported by ONNX/TensorRT/CoreML).
      * ``int8``  → INT8 quantization (ONNX/TensorRT/TFLite); supply ``data``
        for calibration.
      * ``dynamic`` → ONNX dynamic axes.
      * ``simplify`` → simplify the exported ONNX graph.
      * ``workspace`` → TensorRT workspace size (MB).
    """
    from ultralytics import YOLO

    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format {fmt}. Choose from {SUPPORTED_FORMATS}.")

    kwargs: dict = {"format": fmt, "imgsz": imgsz}
    if half:
        kwargs["half"] = True
    if int8:
        kwargs["int8"] = True
    if dynamic and fmt == "onnx":
        kwargs["dynamic"] = True
    if simplify and fmt == "onnx":
        kwargs["simplify"] = True
    if data:
        kwargs["data"] = data
    if workspace and fmt == "engine":
        kwargs["workspace"] = workspace

    model = YOLO(model_path)
    path = model.export(**kwargs)
    path = Path(path)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dst = output_dir / path.name
    if path.resolve() != dst.resolve():
        path.rename(dst)
        path = dst

    metadata = {
        "source_model": str(model_path),
        "format": fmt,
        "imgsz": imgsz,
        "exported_path": str(path),
    }
    (output_dir / "export.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Exported %s to %s", model_path, path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and optimize a trained YOLO model.")
    parser.add_argument("--model", default="models/ppe_yolo11n.pt", help="Model path or name.")
    parser.add_argument("--format", default="onnx", choices=SUPPORTED_FORMATS, help="Export format.")
    parser.add_argument("--imgsz", default=640, type=int, help="Input image size.")
    parser.add_argument("--output", default="outputs/export", help="Output directory.")
    parser.add_argument("--half", action="store_true", help="Use FP16 weights.")
    parser.add_argument("--int8", action="store_true", help="Quantize to INT8 (requires --data for calibration).")
    parser.add_argument("--dynamic", action="store_true", help="ONNX dynamic input axes.")
    parser.add_argument("--simplify", action="store_true", help="Simplify ONNX graph.")
    parser.add_argument("--data", default=None, help="Dataset YAML for INT8 calibration.")
    parser.add_argument("--workspace", default=None, type=int, help="TensorRT workspace size in MB.")
    args = parser.parse_args()
    export(
        args.model,
        args.format,
        args.imgsz,
        args.output,
        half=args.half,
        int8=args.int8,
        dynamic=args.dynamic,
        simplify=args.simplify,
        data=args.data,
        workspace=args.workspace,
    )


if __name__ == "__main__":
    main()
