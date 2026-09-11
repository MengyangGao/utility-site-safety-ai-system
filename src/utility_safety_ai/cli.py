"""Command-line interface for Utility Site Safety AI."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import click
import cv2

from .detection.model_loader import resolve_model_path
from .detection.yolo_detector import YoloDetector
from .events.report import export_report
from .events.store import EventStore
from .events.summary import aggregate_events
from .i18n import SUPPORTED_LANGUAGES, set_language
from .monitoring.profiles import get_monitoring_profile, monitoring_profile_names
from .notifications.delivery import DeliveryWorker, EventHub, configured_senders
from .pipelines.camera_pipeline import run_camera_pipeline
from .pipelines.image_pipeline import run_image_pipeline
from .pipelines.video_pipeline import run_video_pipeline
from .rules.rule_engine import RuleEngine
from .utils.logging import configure_logging
from .utils.paths import generate_run_id
from .zones.zone_loader import load_zones

logger = logging.getLogger(__name__)

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
        click.option(
            "--output", default="outputs", type=click.Path(path_type=Path), help="Output root."
        ),
        click.option(
            "--conf",
            default=None,
            type=click.FloatRange(0.0, 1.0),
            help="Confidence threshold in [0, 1]. Defaults to the selected monitoring profile.",
        ),
        click.option(
            "--iou",
            default=None,
            type=click.FloatRange(0.0, 1.0),
            help="NMS IoU threshold in [0, 1]. Defaults to the selected monitoring profile.",
        ),
        click.option(
            "--profile",
            type=click.Choice(monitoring_profile_names()),
            default="balanced",
            show_default=True,
            help="Monitoring trade-off preset.",
        ),
        click.option("--device", default=None, help="Inference device (cpu, mps, cuda, etc.)."),
        click.option(
            "--blur-faces/--no-blur-faces",
            default=True,
            show_default=True,
            help="Redact privacy-sensitive regions before saving media.",
        ),
        click.option(
            "--privacy-mode",
            type=click.Choice(["gaussian", "pixelate", "solid"]),
            default="gaussian",
            show_default=True,
            help="Privacy redaction style.",
        ),
        click.option(
            "--cooldown",
            default=DEFAULT_COOLDOWN,
            type=click.FloatRange(min=0.0),
            show_default=True,
            help="Event cooldown in seconds.",
        ),
        click.option(
            "--webhook-url",
            default=None,
            envvar="UTILITY_SAFETY_WEBHOOK_URL",
            help="Optional operator-owned HTTP(S) alert endpoint.",
        ),
        click.option("--run-id", default=None, help="Optional stable run identifier."),
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
    conf: float | None,
    iou: float | None,
    profile: str,
    device: str | None,
    blur_faces: bool,
    privacy_mode: str,
    cooldown: float,
    webhook_url: str | None,
    run_id: str | None,
    overwrite: bool,
) -> None:
    """Run safety inference on a single image."""
    effective_run_id = _effective_run_id(run_id)

    def operation():
        selected_profile = get_monitoring_profile(profile)
        zone_list = load_zones(zones)
        detector = YoloDetector(
            model_path=model,
            device=device,
            conf=conf if conf is not None else selected_profile.confidence,
            iou=iou if iou is not None else selected_profile.nms_iou,
        )
        engine = RuleEngine(
            zones=zone_list,
            cooldown_seconds=cooldown,
            rules_config=selected_profile.rule_config(),
        )
        with EventHub(output, configured_senders(webhook_url)) as hub:
            return run_image_pipeline(
                source_path=source,
                output_root=output,
                detector=detector,
                zones=zone_list,
                rule_engine=engine,
                blur_faces_enabled=blur_faces,
                privacy_mode=privacy_mode,
                run_id=effective_run_id,
                overwrite=overwrite,
                monitoring_profile=selected_profile,
                event_sink=hub.publish,
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
@click.option(
    "--max-frames", default=None, type=click.IntRange(min=1), help="Optional frame limit."
)
def infer_video(
    source: Path,
    model: str | None,
    zones: Path | None,
    output: Path,
    conf: float | None,
    iou: float | None,
    profile: str,
    device: str | None,
    blur_faces: bool,
    privacy_mode: str,
    cooldown: float,
    webhook_url: str | None,
    run_id: str | None,
    overwrite: bool,
    max_frames: int | None,
) -> None:
    """Run safety inference on a video file."""
    effective_run_id = _effective_run_id(run_id)

    def operation():
        selected_profile = get_monitoring_profile(profile)
        zone_list = load_zones(zones)
        detector = YoloDetector(
            model_path=model,
            device=device,
            conf=conf if conf is not None else selected_profile.confidence,
            iou=iou if iou is not None else selected_profile.nms_iou,
        )
        engine = RuleEngine(
            zones=zone_list,
            cooldown_seconds=cooldown,
            rules_config=selected_profile.rule_config(),
        )
        with EventHub(output, configured_senders(webhook_url)) as hub:
            return run_video_pipeline(
                source_path=source,
                output_root=output,
                detector=detector,
                zones=zone_list,
                rule_engine=engine,
                blur_faces_enabled=blur_faces,
                privacy_mode=privacy_mode,
                max_frames=max_frames,
                run_id=effective_run_id,
                overwrite=overwrite,
                monitoring_profile=selected_profile,
                event_sink=hub.publish,
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
@click.option(
    "--project", default="runs/train", show_default=True, help="Training project directory."
)
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
@click.option(
    "--source",
    default="0",
    envvar="UTILITY_SAFETY_CAMERA_SOURCE",
    show_default=True,
    help="Camera index, RTSP URL, or device path; supports UTILITY_SAFETY_CAMERA_SOURCE.",
)
@_common_inference_options
@click.option(
    "--duration",
    default=300.0,
    show_default=True,
    type=click.FloatRange(min=0.0, min_open=True),
    help="Optional monotonic runtime limit in seconds.",
)
@click.option(
    "--max-frames", default=None, type=click.IntRange(min=1), help="Optional frame limit."
)
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
    conf: float | None,
    iou: float | None,
    profile: str,
    device: str | None,
    blur_faces: bool,
    privacy_mode: str,
    cooldown: float,
    webhook_url: str | None,
    run_id: str | None,
    overwrite: bool,
    duration: float | None,
    max_frames: int | None,
    display: bool,
) -> None:
    """Run safety inference on a live camera, webcam, or RTSP stream."""
    effective_run_id = _effective_run_id(run_id)

    def operation():
        selected_profile = get_monitoring_profile(profile)
        zone_list = load_zones(zones)
        detector = YoloDetector(
            model_path=model,
            device=device,
            conf=conf if conf is not None else selected_profile.confidence,
            iou=iou if iou is not None else selected_profile.nms_iou,
        )
        engine = RuleEngine(
            zones=zone_list,
            cooldown_seconds=cooldown,
            rules_config=selected_profile.rule_config(),
        )
        with EventHub(output, configured_senders(webhook_url)) as hub:
            return run_camera_pipeline(
                source=source,
                output_root=output,
                detector=detector,
                zones=zone_list,
                rule_engine=engine,
                blur_faces_enabled=blur_faces,
                privacy_mode=privacy_mode,
                duration_seconds=duration,
                max_frames=max_frames,
                display=display,
                run_id=effective_run_id,
                overwrite=overwrite,
                monitoring_profile=selected_profile,
                event_sink=hub.publish,
            )

    events = _run_checked(operation)
    _echo_result(events, output / "runs" / effective_run_id, prefix="Camera inference complete")


@main.command("doctor")
@click.option(
    "--model", default=None, help="Optional model path/name to validate without loading it."
)
def doctor(model: str | None) -> None:
    """Report environment, acceleration, model and video-codec readiness."""

    import platform

    import torch

    resolved = resolve_model_path(model)
    model_path = Path(resolved)
    model_state = "local" if model_path.is_file() else "runtime-download"
    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "opencv": cv2.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        "model": str(resolved),
        "model_state": model_state,
        "privacy_face_cascade": bool(getattr(cv2, "data", None)),
    }
    click.echo(json.dumps(payload, indent=2))


@main.command("model-gate", hidden=True)
@click.option(
    "--metrics",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
)
@click.option("--min-precision", default=0.65, show_default=True, type=click.FloatRange(0, 1))
@click.option("--min-recall", default=0.65, show_default=True, type=click.FloatRange(0, 1))
@click.option("--min-map50", default=0.60, show_default=True, type=click.FloatRange(0, 1))
@click.option("--min-class-recall", default=0.50, show_default=True, type=click.FloatRange(0, 1))
@click.option("--required-class", "required_classes", multiple=True)
def model_gate(
    metrics: Path,
    min_precision: float,
    min_recall: float,
    min_map50: float,
    min_class_recall: float,
    required_classes: tuple[str, ...],
) -> None:
    """Validate model metrics against configurable thresholds."""

    from .training.quality_gate import assess_model_metrics

    report = json.loads(metrics.read_text(encoding="utf-8"))
    result = assess_model_metrics(
        report,
        min_precision=min_precision,
        min_recall=min_recall,
        min_map50=min_map50,
        min_class_recall=min_class_recall,
        required_classes=required_classes,
    )
    click.echo(json.dumps(result, indent=2))
    if not result["passed"]:
        raise click.ClickException("Model metrics did not meet the configured thresholds")


@main.command("audit-provenance", hidden=True)
@click.option(
    "--manifest",
    default="docs/legal/provenance.yaml",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    show_default=True,
    help="Repository provenance manifest.",
)
@click.option(
    "--repo-root",
    default=".",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    show_default=True,
)
def audit_provenance_cmd(manifest: Path, repo_root: Path) -> None:
    """Reject unregistered artifacts, unapproved licenses, and hash drift."""
    from .governance import audit_provenance

    report = audit_provenance(manifest, repo_root.resolve())
    click.echo(json.dumps(report, indent=2))
    if not report["passed"]:
        raise click.ClickException("Provenance audit failed")


@main.command("web")
def web() -> None:
    """Launch the Streamlit monitoring console."""
    app_path = Path(__file__).resolve().parent / "web" / "app.py"
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.address=127.0.0.1",
        "--browser.gatherUsageStats=false",
        "--theme.base=dark",
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise click.ClickException(f"Streamlit exited with status {exc.returncode}") from exc


@main.command("export-report")
@click.option(
    "--events",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    help="Events JSONL file.",
)
@click.option(
    "--output",
    required=True,
    type=click.Path(path_type=Path),
    help="Output report path or directory.",
)
@click.option(
    "--format", "fmt", default="csv", type=click.Choice(["csv", "jsonl"]), show_default=True
)
def export_report_cmd(events: Path, output: Path, fmt: str) -> None:
    """Export an events JSONL file to CSV or JSONL."""
    path = _run_checked(lambda: export_report(events, output, fmt=fmt))
    click.echo(f"Report exported to {path}")


@main.command("fetch-model")
@click.option(
    "--model", default="yolo11n.pt", show_default=True, help="Ultralytics model asset name."
)
@click.option("--output", default="models", type=click.Path(path_type=Path), show_default=True)
@click.option("--force", is_flag=True, help="Replace an existing destination model.")
def fetch_model(model: str, output: Path, force: bool) -> None:
    """Download and pin a small Ultralytics model with a SHA-256 record."""

    def operation() -> tuple[Path, str]:
        from importlib.metadata import version

        from .detection.runtime import load_yolo

        ultralytics_version = version("ultralytics")

        output.mkdir(parents=True, exist_ok=True)
        requested = Path(model)
        destination = output / requested.name
        source_existed_before = requested.is_file()
        requested_is_destination = (
            source_existed_before and requested.resolve() == destination.resolve()
        )
        if destination.exists() and not requested_is_destination and not force:
            raise FileExistsError(
                f"Model already exists: {destination}. Use --force to replace it."
            )

        loaded = load_yolo(model, allow_download=True)
        source = Path(str(getattr(loaded, "ckpt_path", model)))
        if not source.is_file():
            source = Path(model)
        if not source.is_file():
            raise FileNotFoundError(
                f"Ultralytics loaded the model but its checkpoint was not found: {model}"
            )
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
@click.option(
    "--output", default="outputs/export", type=click.Path(path_type=Path), show_default=True
)
def export_model(model: str, fmt: str, imgsz: int, device: str | None, output: Path) -> None:
    """Export a YOLO checkpoint to a deployment format."""

    def operation() -> Path:
        from .tools.export_model import export

        return export(model, fmt, imgsz, output, device=device)

    destination = _run_checked(operation)
    click.echo(f"Model exported to {destination}")


@main.command("api")
@click.option("--output", default="outputs", type=click.Path(path_type=Path), show_default=True)
@click.option("--port", default=8080, type=click.IntRange(1, 65535), show_default=True)
def api_command(output: Path, port: int) -> None:
    """Serve authenticated local events, evidence and review endpoints (install .[api])."""

    def operation():
        import uvicorn

        from .api import create_app

        uvicorn.run(create_app(output), host="127.0.0.1", port=port)

    _run_checked(operation)


@main.command("deliver-pending")
@click.option("--output", default="outputs", type=click.Path(path_type=Path), show_default=True)
@click.option("--retry-failed", is_flag=True, help="Explicitly requeue dead-letter deliveries.")
def deliver_pending(output: Path, retry_failed: bool) -> None:
    """Retry due outbox entries using currently configured webhook/MQTT endpoints."""

    def operation():
        store = EventStore(output / "events.sqlite3")
        senders = configured_senders()
        if not senders:
            raise ValueError("Configure a webhook or MQTT endpoint before retrying deliveries.")
        if retry_failed:
            store.retry_failed()
        worker = DeliveryWorker(store, senders)
        while worker.run_once():
            pass
        return store.delivery_summary()

    click.echo(json.dumps(_run_checked(operation), indent=2))


if __name__ == "__main__":
    main()
