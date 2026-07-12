"""Command-line interface for Utility Site Safety AI."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import click

from .detection.yolo_detector import YoloDetector
from .events.report import export_report
from .events.summary import aggregate_events
from .i18n import SUPPORTED_LANGUAGES, set_language
from .pipelines.camera_pipeline import run_camera_pipeline
from .pipelines.image_pipeline import run_image_pipeline
from .pipelines.video_pipeline import run_video_pipeline
from .rules.rule_engine import RuleEngine
from .utils.logging import configure_logging
from .utils.paths import generate_run_id
from .zones.zone_loader import load_zones

logger = logging.getLogger(__name__)

DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
DEFAULT_COOLDOWN = 10.0
T = TypeVar("T")


def _set_language(ctx: click.Context, param: click.Parameter, value: str) -> None:
    """Click callback that applies the selected language before commands run."""
    del ctx, param
    if value:
        set_language(value)


def _run_checked(operation: Callable[[], T]) -> T:
    """Render expected runtime failures as concise, non-zero CLI errors."""
    try:
        return operation()
    except click.ClickException:
        raise
    except Exception as exc:
        logger.debug("Command failed", exc_info=True)
        raise click.ClickException(str(exc)) from exc


def _effective_run_id(run_id: str | None) -> str:
    return run_id or generate_run_id()


def _echo_result(events, run_dir: Path, *, prefix: str = "Inference complete") -> None:
    summary = aggregate_events(events)
    click.echo(
        f"{prefix}. {len(events)} event(s): "
        f"{summary.get('zone_intrusions', 0)} zone intrusion(s), "
        f"{summary.get('ppe_violations', 0)} PPE violation(s)."
    )
    click.echo(f"Run artifacts: {run_dir}")


@click.group()
@click.option("--verbose", is_flag=True, help="Enable debug logging.")
@click.option(
    "--lang",
    "language",
    default="en",
    type=click.Choice(sorted(SUPPORTED_LANGUAGES), case_sensitive=False),
    callback=_set_language,
    is_eager=True,
    help="UI and annotation language (en, zh-hans, zh-hant).",
)
def main(verbose: bool, language: str) -> None:
    """Utility Site Safety AI System — CLI."""
    del language
    configure_logging(level=logging.DEBUG if verbose else logging.INFO)


def _common_inference_options(function):
    options = [
        click.option("--model", default=None, help="YOLO model path or name."),
        click.option(
            "--zones",
            default=None,
            type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
            help="Zone YAML/JSON file. An explicitly supplied path must be valid.",
        ),
        click.option("--output", default="outputs", type=click.Path(path_type=Path), help="Output root."),
        click.option(
            "--conf",
            default=DEFAULT_CONF,
            type=click.FloatRange(0.0, 1.0),
            show_default=True,
            help="Confidence threshold in [0, 1].",
        ),
        click.option(
            "--iou",
            default=DEFAULT_IOU,
            type=click.FloatRange(0.0, 1.0),
            show_default=True,
            help="NMS IoU threshold in [0, 1].",
        ),
        click.option("--device", default=None, help="Inference device (cpu, mps, cuda, etc.)."),
        click.option("--blur-faces", is_flag=True, help="Blur privacy-sensitive regions."),
        click.option(
            "--cooldown",
            default=DEFAULT_COOLDOWN,
            type=click.FloatRange(min=0.0),
            show_default=True,
            help="Event cooldown in seconds.",
        ),
        click.option("--run-id", default=None, help="Optional deterministic audit run identifier."),
        click.option(
            "--overwrite",
            is_flag=True,
            help="Explicitly replace an existing run with the same --run-id.",
        ),
    ]
    for option in reversed(options):
        function = option(function)
    return function


@main.command()
@click.option(
    "--source",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    help="Input image path.",
)
@_common_inference_options
def infer_image(
    source: Path,
    model: str | None,
    zones: Path | None,
    output: Path,
    conf: float,
    iou: float,
    device: str | None,
    blur_faces: bool,
    cooldown: float,
    run_id: str | None,
    overwrite: bool,
) -> None:
    """Run safety inference on a single image."""
    effective_run_id = _effective_run_id(run_id)

    def operation():
        zone_list = load_zones(zones)
        detector = YoloDetector(model_path=model, device=device, conf=conf, iou=iou)
        engine = RuleEngine(zones=zone_list, cooldown_seconds=cooldown)
        return run_image_pipeline(
            source_path=source,
            output_root=output,
            detector=detector,
            zones=zone_list,
            rule_engine=engine,
            blur_faces_enabled=blur_faces,
            run_id=effective_run_id,
            overwrite=overwrite,
        )

    _, events = _run_checked(operation)
    _echo_result(events, output / "runs" / effective_run_id)


@main.command()
@click.option(
    "--source",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    help="Input video path.",
)
@_common_inference_options
@click.option("--max-frames", default=None, type=click.IntRange(min=1), help="Optional frame limit.")
def infer_video(
    source: Path,
    model: str | None,
    zones: Path | None,
    output: Path,
    conf: float,
    iou: float,
    device: str | None,
    blur_faces: bool,
    cooldown: float,
    run_id: str | None,
    overwrite: bool,
    max_frames: int | None,
) -> None:
    """Run safety inference on a video file."""
    effective_run_id = _effective_run_id(run_id)

    def operation():
        zone_list = load_zones(zones)
        detector = YoloDetector(model_path=model, device=device, conf=conf, iou=iou)
        engine = RuleEngine(zones=zone_list, cooldown_seconds=cooldown)
        return run_video_pipeline(
            source_path=source,
            output_root=output,
            detector=detector,
            zones=zone_list,
            rule_engine=engine,
            blur_faces_enabled=blur_faces,
            max_frames=max_frames,
            run_id=effective_run_id,
            overwrite=overwrite,
        )

    events = _run_checked(operation)
    _echo_result(events, output / "runs" / effective_run_id)


@main.command()
@click.option("--data", required=True, help="Dataset YAML path or Ultralytics dataset name.")
@click.option("--model", default="yolo11n.pt", show_default=True, help="Base model name or path.")
@click.option("--epochs", default=30, type=click.IntRange(min=1), show_default=True)
@click.option("--imgsz", default=640, type=click.IntRange(min=32), show_default=True)
@click.option("--batch", default=16, type=click.IntRange(min=1), show_default=True)
@click.option("--device", default=None, help="Training device.")
@click.option("--project", default="runs/train", show_default=True, help="Training project directory.")
@click.option("--name", default="ppe", show_default=True, help="Training run name.")
@click.option(
    "--resume",
    is_flag=True,
    help="Resume state from the existing checkpoint supplied with --model.",
)
def train(
    data: str,
    model: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str | None,
    project: str,
    name: str,
    resume: bool,
) -> None:
    """Fine-tune a YOLO model on a PPE dataset."""

    def operation() -> None:
        from .training.train_yolo import train as train_yolo

        train_yolo(
            data=data,
            model=model,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=project,
            name=name,
            resume=resume,
        )

    _run_checked(operation)
    click.echo("Training complete.")


@main.command("infer-camera")
@click.option("--source", default="0", show_default=True, help="Camera index, RTSP URL, or device path.")
@_common_inference_options
@click.option(
    "--duration",
    default=None,
    type=click.FloatRange(min=0.0, min_open=True),
    help="Optional monotonic runtime limit in seconds.",
)
@click.option("--max-frames", default=None, type=click.IntRange(min=1), help="Optional frame limit.")
@click.option(
    "--display",
    is_flag=True,
    help="Show an OpenCV preview only when a GUI-enabled OpenCV build is installed.",
)
def infer_camera(
    source: str,
    model: str | None,
    zones: Path | None,
    output: Path,
    conf: float,
    iou: float,
    device: str | None,
    blur_faces: bool,
    cooldown: float,
    run_id: str | None,
    overwrite: bool,
    duration: float | None,
    max_frames: int | None,
    display: bool,
) -> None:
    """Run safety inference on a live camera, webcam, or RTSP stream."""
    effective_run_id = _effective_run_id(run_id)

    def operation():
        zone_list = load_zones(zones)
        detector = YoloDetector(model_path=model, device=device, conf=conf, iou=iou)
        engine = RuleEngine(zones=zone_list, cooldown_seconds=cooldown)
        return run_camera_pipeline(
            source=source,
            output_root=output,
            detector=detector,
            zones=zone_list,
            rule_engine=engine,
            blur_faces_enabled=blur_faces,
            duration_seconds=duration,
            max_frames=max_frames,
            display=display,
            run_id=effective_run_id,
            overwrite=overwrite,
        )

    events = _run_checked(operation)
    _echo_result(events, output / "runs" / effective_run_id, prefix="Camera inference complete")


@main.command("export-report")
@click.option(
    "--events",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    help="Events JSONL file.",
)
@click.option("--output", required=True, type=click.Path(path_type=Path), help="Output report path or directory.")
@click.option("--format", "fmt", default="csv", type=click.Choice(["csv", "jsonl"]), show_default=True)
def export_report_cmd(events: Path, output: Path, fmt: str) -> None:
    """Export an events JSONL file to CSV or JSONL."""
    path = _run_checked(lambda: export_report(events, output, fmt=fmt))
    click.echo(f"Report exported to {path}")


@main.command("fetch-model")
@click.option("--model", default="yolo11n.pt", show_default=True, help="Ultralytics model asset name.")
@click.option("--output", default="models", type=click.Path(path_type=Path), show_default=True)
@click.option("--force", is_flag=True, help="Replace an existing destination model.")
def fetch_model(model: str, output: Path, force: bool) -> None:
    """Download and pin a small Ultralytics model with a SHA-256 record."""

    def operation() -> tuple[Path, str]:
        from ultralytics import YOLO
        from ultralytics import __version__ as ultralytics_version

        output.mkdir(parents=True, exist_ok=True)
        requested = Path(model)
        destination = output / requested.name
        source_existed_before = requested.is_file()
        requested_is_destination = (
            source_existed_before
            and requested.resolve() == destination.resolve()
        )
        if destination.exists() and not requested_is_destination and not force:
            raise FileExistsError(
                f"Model already exists: {destination}. Use --force to replace it."
            )

        loaded = YOLO(model)
        source = Path(str(getattr(loaded, "ckpt_path", model)))
        if not source.is_file():
            source = Path(model)
        if not source.is_file():
            raise FileNotFoundError(f"Ultralytics loaded the model but its checkpoint was not found: {model}")
        if source.resolve() != destination.resolve():
            auto_downloaded_in_cwd = (
                not source_existed_before
                and requested.parent == Path(".")
                and source.resolve() == (Path.cwd() / requested.name).resolve()
            )
            file_descriptor, temporary_name = tempfile.mkstemp(
                dir=output,
                prefix=f".{destination.name}.",
                suffix=".tmp",
            )
            os.close(file_descriptor)
            temporary = Path(temporary_name)
            try:
                shutil.copy2(source, temporary)
                if temporary.stat().st_size == 0:
                    raise OSError(f"Downloaded model is empty: {source}")
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
            if auto_downloaded_in_cwd:
                source.unlink(missing_ok=True)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        metadata = {
            "schema_version": "1.0",
            "model": model,
            "path": str(destination),
            "sha256": digest,
            "ultralytics_version": ultralytics_version,
        }
        destination.with_suffix(destination.suffix + ".json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        return destination, digest

    destination, digest = _run_checked(operation)
    click.echo(f"Model ready: {destination}")
    click.echo(f"SHA-256: {digest}")


@main.command("export-model")
@click.option("--model", required=True, help="Trained YOLO model path or name.")
@click.option(
    "--format",
    "fmt",
    default="onnx",
    type=click.Choice(["onnx", "torchscript", "openvino", "ncnn", "coreml", "tflite", "engine"]),
    show_default=True,
)
@click.option("--imgsz", default=640, type=click.IntRange(min=32), show_default=True)
@click.option("--device", default=None, help="Export device.")
@click.option("--output", default="outputs/export", type=click.Path(path_type=Path), show_default=True)
def export_model(model: str, fmt: str, imgsz: int, device: str | None, output: Path) -> None:
    """Export a YOLO checkpoint to a deployment format."""

    def operation() -> Path:
        from ultralytics import YOLO

        kwargs = {"format": fmt, "imgsz": imgsz}
        if device is not None:
            kwargs["device"] = device
        exported = Path(str(YOLO(model).export(**kwargs)))
        if not exported.is_file():
            raise OSError(f"Model export did not produce a file: {exported}")
        output.mkdir(parents=True, exist_ok=True)
        destination = output / exported.name
        if exported.resolve() != destination.resolve():
            shutil.copy2(exported, destination)
        return destination

    destination = _run_checked(operation)
    click.echo(f"Model exported to {destination}")


if __name__ == "__main__":
    main()
