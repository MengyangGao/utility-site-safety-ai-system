"""Command-line interface for Utility Site Safety AI."""

from __future__ import annotations

import logging

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
from .zones.zone_loader import load_zones

logger = logging.getLogger(__name__)

DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
DEFAULT_COOLDOWN = 10.0


def _set_language(ctx: click.Context, param: click.Parameter, value: str) -> None:
    """Click callback that applies the selected language before commands run."""
    del ctx, param
    if value:
        set_language(value)


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


@main.command()
@click.option("--source", required=True, type=click.Path(exists=True), help="Input image path.")
@click.option("--model", default=None, help="YOLO model path or name.")
@click.option("--zones", default=None, type=click.Path(exists=False), help="Zone YAML/JSON file.")
@click.option("--output", default="outputs/infer-image", help="Output directory.")
@click.option("--conf", default=DEFAULT_CONF, type=float, help="Confidence threshold.")
@click.option("--iou", default=DEFAULT_IOU, type=float, help="NMS IoU threshold.")
@click.option("--device", default=None, help="Inference device (cpu, mps, cuda, etc.).")
@click.option("--blur-faces", is_flag=True, help="Blur privacy-sensitive regions.")
@click.option("--cooldown", default=DEFAULT_COOLDOWN, type=float, help="Event cooldown in seconds.")
def infer_image(
    source: str,
    model: str | None,
    zones: str | None,
    output: str,
    conf: float,
    iou: float,
    device: str | None,
    blur_faces: bool,
    cooldown: float,
) -> None:
    """Run safety inference on a single image."""
    zone_list = load_zones(zones)
    detector = YoloDetector(model_path=model, device=device, conf=conf, iou=iou)
    engine = RuleEngine(zones=zone_list, cooldown_seconds=cooldown)
    _, events = run_image_pipeline(
        source_path=source,
        output_root=output,
        detector=detector,
        zones=zone_list,
        rule_engine=engine,
        blur_faces_enabled=blur_faces,
    )
    summary = aggregate_events(events)
    click.echo(
        f"Inference complete. {len(events)} event(s): "
        f"{summary.get('zone_intrusions', 0)} zone intrusion(s), "
        f"{summary.get('ppe_violations', 0)} PPE violation(s)."
    )


@main.command()
@click.option("--source", required=True, type=click.Path(exists=True), help="Input video path.")
@click.option("--model", default=None, help="YOLO model path or name.")
@click.option("--zones", default=None, type=click.Path(exists=False), help="Zone YAML/JSON file.")
@click.option("--output", default="outputs/infer-video", help="Output directory.")
@click.option("--conf", default=DEFAULT_CONF, type=float, help="Confidence threshold.")
@click.option("--iou", default=DEFAULT_IOU, type=float, help="NMS IoU threshold.")
@click.option("--device", default=None, help="Inference device (cpu, mps, cuda, etc.).")
@click.option("--blur-faces", is_flag=True, help="Blur privacy-sensitive regions.")
@click.option("--cooldown", default=DEFAULT_COOLDOWN, type=float, help="Event cooldown in seconds.")
@click.option("--max-frames", default=None, type=int, help="Optional frame limit.")
def infer_video(
    source: str,
    model: str | None,
    zones: str | None,
    output: str,
    conf: float,
    iou: float,
    device: str | None,
    blur_faces: bool,
    cooldown: float,
    max_frames: int | None,
) -> None:
    """Run safety inference on a video file."""
    zone_list = load_zones(zones)
    detector = YoloDetector(model_path=model, device=device, conf=conf, iou=iou)
    engine = RuleEngine(zones=zone_list, cooldown_seconds=cooldown)
    events = run_video_pipeline(
        source_path=source,
        output_root=output,
        detector=detector,
        zones=zone_list,
        rule_engine=engine,
        blur_faces_enabled=blur_faces,
        max_frames=max_frames,
    )
    summary = aggregate_events(events)
    click.echo(
        f"Inference complete. {len(events)} event(s): "
        f"{summary.get('zone_intrusions', 0)} zone intrusion(s), "
        f"{summary.get('ppe_violations', 0)} PPE violation(s)."
    )


@main.command()
@click.option("--data", required=True, help="Dataset YAML path or Ultralytics dataset name (e.g. construction-ppe.yaml).")
@click.option("--model", default="yolo11n.pt", help="Base model name or path.")
@click.option("--epochs", default=30, type=int)
@click.option("--imgsz", default=640, type=int)
@click.option("--batch", default=16, type=int)
@click.option("--device", default=None, help="Training device.")
@click.option("--project", default="runs/train", help="Training project directory.")
@click.option("--name", default="ppe", help="Training run name.")
@click.option("--resume", is_flag=True, help="Resume from the last checkpoint of the named run.")
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
    from .training.train_yolo import train as _train

    _train(
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
    click.echo("Training complete.")


@main.command("infer-camera")
@click.option("--source", default="0", help="Camera index, RTSP URL, or camera device path.")
@click.option("--model", default=None, help="YOLO model path or name.")
@click.option("--zones", default=None, type=click.Path(exists=False), help="Zone YAML/JSON file.")
@click.option("--output", default="outputs/infer-camera", help="Output directory.")
@click.option("--conf", default=DEFAULT_CONF, type=float, help="Confidence threshold.")
@click.option("--iou", default=DEFAULT_IOU, type=float, help="NMS IoU threshold.")
@click.option("--device", default=None, help="Inference device (cpu, mps, cuda, etc.).")
@click.option("--blur-faces", is_flag=True, help="Blur privacy-sensitive regions.")
@click.option("--cooldown", default=DEFAULT_COOLDOWN, type=float, help="Event cooldown in seconds.")
@click.option("--duration", default=None, type=float, help="Optional runtime limit in seconds.")
@click.option("--max-frames", default=None, type=int, help="Optional frame limit.")
@click.option("--display", is_flag=True, help="Show OpenCV preview window (requires a display).")
def infer_camera(
    source: str,
    model: str | None,
    zones: str | None,
    output: str,
    conf: float,
    iou: float,
    device: str | None,
    blur_faces: bool,
    cooldown: float,
    duration: float | None,
    max_frames: int | None,
    display: bool,
) -> None:
    """Run safety inference on a live camera, webcam, or RTSP stream."""
    zone_list = load_zones(zones)
    detector = YoloDetector(model_path=model, device=device, conf=conf, iou=iou)
    engine = RuleEngine(zones=zone_list, cooldown_seconds=cooldown)
    events = run_camera_pipeline(
        source=source,
        output_root=output,
        detector=detector,
        zones=zone_list,
        rule_engine=engine,
        blur_faces_enabled=blur_faces,
        duration_seconds=duration,
        max_frames=max_frames,
        display=display,
    )
    summary = aggregate_events(events)
    click.echo(
        f"Camera inference complete. {len(events)} event(s): "
        f"{summary.get('zone_intrusions', 0)} zone intrusion(s), "
        f"{summary.get('ppe_violations', 0)} PPE violation(s)."
    )


@main.command()
@click.option("--events", required=True, type=click.Path(exists=True), help="Events JSONL file.")
@click.option("--output", required=True, help="Output report path or directory.")
@click.option("--format", "fmt", default="csv", type=click.Choice(["csv", "jsonl"]))
def export_report_cmd(events: str, output: str, fmt: str) -> None:
    """Export an events JSONL file to CSV or JSONL."""
    path = export_report(events, output, fmt=fmt)
    click.echo(f"Report exported to {path}")


if __name__ == "__main__":
    main()
