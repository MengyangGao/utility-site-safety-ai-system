"""Pure helpers for the Streamlit safety-monitoring experience.

The functions in this module deliberately avoid importing Streamlit or
Ultralytics.  That keeps zone editing, credential redaction, model inspection,
and run-history shaping deterministic and easy to unit test.
"""

from __future__ import annotations

import hashlib
import math
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from .pipelines._artifacts import redact_source
from .zones.zone import PPE_TYPES, Zone
from .zones.zone_loader import _ROOT_FIELDS as _CORE_ROOT_FIELDS
from .zones.zone_loader import _ZONE_FIELDS as _CORE_ZONE_FIELDS

RISK_LEVELS = ("low", "medium", "high", "critical")
PPE_CLASS_NAMES = {
    "helmet",
    "vest",
    "gloves",
    "boots",
    "goggles",
    "no_helmet",
    "no_vest",
    "no_gloves",
    "no_boots",
    "no_goggles",
    "no_goggle",
}


class ZoneValidationError(ValueError):
    """Raised when an interactive zone configuration is unsafe or ambiguous."""


@dataclass(frozen=True)
class NormalizedZone:
    """A restricted zone whose polygon is expressed in the ``[0, 1]`` range."""

    id: str
    name: str
    risk_level: str
    polygon: tuple[tuple[float, float], ...]
    required_ppe: tuple[str, ...] = ()
    dwell_seconds: float = 0.0


@dataclass
class RunContext:
    """Session-owned inference state that must survive Streamlit reruns."""

    context_id: str
    output_root: Path
    detector: Any
    engine: Any
    tracker: Any
    model_info: dict[str, Any]
    events: list[Any] = field(default_factory=list)


def new_session_id() -> str:
    """Return an opaque identifier suitable for a private Web output root."""

    return uuid.uuid4().hex


def utc_run_label() -> str:
    """Return a compact, sortable UTC label for UI-only records."""

    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def parse_zone_yaml(
    text: str,
    *,
    source_width: int | None = None,
    source_height: int | None = None,
) -> list[NormalizedZone]:
    """Parse strict YAML into normalized zones.

    Each zone must declare ``coordinate_space`` when coordinates are normalized.
    Legacy pixel presets are accepted only when their source dimensions are
    supplied, preventing a zone created for one resolution from silently being
    applied to another. Field names match the strict core CLI schema.
    """

    try:
        data = yaml.safe_load(text) if text.strip() else {"zones": []}
    except yaml.YAMLError as exc:
        raise ZoneValidationError(f"YAML syntax error: {exc}") from exc

    if not isinstance(data, dict):
        raise ZoneValidationError("The YAML root must be a mapping containing 'zones'.")
    unknown_root_fields = set(data) - _CORE_ROOT_FIELDS
    if unknown_root_fields:
        raise ZoneValidationError(
            f"Unknown zone configuration fields: {sorted(unknown_root_fields)}."
        )
    raw_zones = data.get("zones", [])
    if raw_zones is None:
        raw_zones = []
    if not isinstance(raw_zones, list):
        raise ZoneValidationError("'zones' must be a list.")

    result: list[NormalizedZone] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_zones, start=1):
        if not isinstance(raw, dict):
            raise ZoneValidationError(f"Zone #{index} must be a mapping.")
        unknown_zone_fields = set(raw) - _CORE_ZONE_FIELDS
        if unknown_zone_fields:
            raise ZoneValidationError(
                f"Zone #{index} has unknown fields: {sorted(unknown_zone_fields)}."
            )
        zone_id = str(raw.get("id", "")).strip()
        if not zone_id:
            raise ZoneValidationError(f"Zone #{index} requires a non-empty id.")
        if zone_id in seen_ids:
            raise ZoneValidationError(f"Duplicate zone id: {zone_id!r}.")
        seen_ids.add(zone_id)

        name = str(raw.get("name", zone_id)).strip() or zone_id
        risk_level = str(raw.get("risk_level", "high")).strip().lower()
        if risk_level not in RISK_LEVELS:
            raise ZoneValidationError(
                f"Zone {zone_id!r} has invalid risk_level {risk_level!r}; "
                f"choose one of {', '.join(RISK_LEVELS)}."
            )

        polygon = _parse_polygon(raw.get("polygon"), zone_id)
        item_space = str(raw.get("coordinate_space", "auto")).strip().lower()
        if item_space == "auto":
            item_space = "normalized" if _looks_normalized(polygon) else "pixel"
        if item_space in {"pixel", "pixels"}:
            if not source_width or not source_height:
                raise ZoneValidationError(
                    f"Zone {zone_id!r} uses pixel coordinates; source_width and "
                    "source_height are required to normalize it."
                )
            polygon = tuple((x / source_width, y / source_height) for x, y in polygon)
        elif item_space != "normalized":
            raise ZoneValidationError(
                f"Zone {zone_id!r} has unsupported coordinate_space {item_space!r}."
            )

        _validate_normalized_polygon(polygon, zone_id)
        required_ppe = _parse_required_ppe(raw.get("required_ppe", []), zone_id)
        dwell_seconds = _parse_dwell_seconds(raw.get("dwell_seconds", 0.0), zone_id)
        result.append(
            NormalizedZone(
                zone_id,
                name,
                risk_level,
                polygon,
                required_ppe,
                dwell_seconds,
            )
        )
    return result


def zones_from_rows(rows: list[dict[str, Any]]) -> list[NormalizedZone]:
    """Parse editable table rows into strict normalized zones."""

    raw_zones: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if not any(str(value).strip() for value in row.values() if value is not None):
            continue
        polygon_text = str(row.get("polygon", "")).strip()
        points: list[list[float]] = []
        try:
            for token in polygon_text.split(";"):
                if not token.strip():
                    continue
                pair = [float(value.strip()) for value in token.split(",")]
                if len(pair) != 2:
                    raise ValueError
                points.append(pair)
        except (TypeError, ValueError) as exc:
            raise ZoneValidationError(
                f"Zone row #{index} polygon must use 'x,y; x,y; x,y' normalized coordinates."
            ) from exc
        required_ppe_value = row.get("required_ppe", "")
        if required_ppe_value is None or required_ppe_value == "":
            required_ppe: list[str] = []
        elif isinstance(required_ppe_value, str):
            required_ppe = [
                value.strip()
                for value in required_ppe_value.split(",")
                if value.strip()
            ]
        elif isinstance(required_ppe_value, (list, tuple)):
            required_ppe = list(required_ppe_value)
        else:
            raise ZoneValidationError(
                f"Zone row #{index} required_ppe must be a comma-separated list."
            )

        dwell_value = row.get("dwell_seconds", 0.0)
        if dwell_value is None or dwell_value == "":
            dwell_seconds: Any = 0.0
        else:
            dwell_seconds = dwell_value
        raw_zones.append(
            {
                "id": row.get("id", ""),
                "name": row.get("name", ""),
                "risk_level": row.get("risk_level", "high"),
                "polygon": points,
                "required_ppe": required_ppe,
                "dwell_seconds": dwell_seconds,
                "coordinate_space": "normalized",
            }
        )
    return parse_zone_yaml(
        yaml.safe_dump(
            {"zones": raw_zones},
            sort_keys=False,
        )
    )


def zones_to_rows(zones: list[NormalizedZone]) -> list[dict[str, Any]]:
    """Serialize normalized zones for ``st.data_editor``."""

    rows: list[dict[str, Any]] = []
    for zone in zones:
        polygon = "; ".join(f"{x:.4f},{y:.4f}" for x, y in zone.polygon)
        rows.append(
            {
                "id": zone.id,
                "name": zone.name,
                "risk_level": zone.risk_level,
                "required_ppe": ", ".join(zone.required_ppe),
                "dwell_seconds": zone.dwell_seconds,
                "polygon": polygon,
            }
        )
    return rows


def zones_to_yaml(zones: list[NormalizedZone]) -> str:
    """Serialize zones in the portable normalized-coordinate schema."""

    data = {
        "zones": [
            {
                "id": zone.id,
                "name": zone.name,
                "risk_level": zone.risk_level,
                "coordinate_space": "normalized",
                "required_ppe": list(zone.required_ppe),
                "dwell_seconds": zone.dwell_seconds,
                "polygon": [[round(x, 6), round(y, 6)] for x, y in zone.polygon],
            }
            for zone in zones
        ],
    }
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def zones_to_pixels(
    zones: list[NormalizedZone], width: int, height: int
) -> list[Zone]:
    """Scale portable normalized zones to a concrete source resolution."""

    if width <= 0 or height <= 0:
        raise ZoneValidationError("Source width and height must be positive.")
    return [
        Zone(
            id=zone.id,
            name=zone.name,
            risk_level=zone.risk_level,
            polygon=[(x * width, y * height) for x, y in zone.polygon],
            required_ppe=zone.required_ppe,
            dwell_seconds=zone.dwell_seconds,
        )
        for zone in zones
    ]


def zones_to_domain(zones: list[NormalizedZone]) -> list[Zone]:
    """Convert editor zones to normalized core ``Zone`` objects."""

    return [
        Zone(
            id=zone.id,
            name=zone.name,
            risk_level=zone.risk_level,
            polygon=list(zone.polygon),
            coordinate_space="normalized",
            required_ppe=zone.required_ppe,
            dwell_seconds=zone.dwell_seconds,
        )
        for zone in zones
    ]


def draw_zone_preview(
    zones: list[NormalizedZone],
    image: np.ndarray | None = None,
    *,
    width: int = 800,
    height: int = 450,
) -> np.ndarray:
    """Draw normalized zones over an image or an industrial-style blank canvas."""

    if image is None:
        canvas = np.full((height, width, 3), (28, 35, 48), dtype=np.uint8)
        for x in range(0, width, 50):
            cv2.line(canvas, (x, 0), (x, height), (48, 58, 74), 1)
        for y in range(0, height, 50):
            cv2.line(canvas, (0, y), (width, y), (48, 58, 74), 1)
    else:
        canvas = image.copy()
        height, width = canvas.shape[:2]

    overlay = canvas.copy()
    colors = {
        "low": (52, 168, 83),
        "medium": (20, 190, 235),
        "high": (30, 120, 245),
        "critical": (45, 45, 220),
    }
    for zone in zones:
        points = np.array(
            [[round(x * width), round(y * height)] for x, y in zone.polygon],
            dtype=np.int32,
        )
        color = colors[zone.risk_level]
        cv2.fillPoly(overlay, [points], color)
        cv2.polylines(canvas, [points], True, color, 3)
        anchor = tuple(points[0])
        cv2.putText(
            canvas,
            f"{zone.name} [{zone.risk_level}]",
            (int(anchor[0]), max(20, int(anchor[1]) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0)


def redact_uri_credentials(source: str) -> str:
    """Apply the same URL-secret policy used by all pipeline manifests."""

    return redact_source(source)


def sanitize_run_artifacts(
    run_dir: str | Path, raw_source: str, redacted_source: str | None = None
) -> int:
    """Replace a credential-bearing source string in downloadable text artifacts."""

    run_dir = Path(run_dir)
    redacted_source = redacted_source or redact_uri_credentials(raw_source)
    if raw_source == redacted_source or not run_dir.exists():
        return 0
    changed = 0
    for pattern in ("*.json", "*.jsonl", "*.csv"):
        for path in run_dir.rglob(pattern):
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if raw_source not in text:
                continue
            path.write_text(text.replace(raw_source, redacted_source), encoding="utf-8")
            changed += 1
    return changed


@contextmanager
def temporary_upload(uploaded_file: Any) -> Iterator[Path]:
    """Persist a Streamlit upload for a pipeline call and always remove it."""

    import tempfile

    suffix = Path(getattr(uploaded_file, "name", "upload.bin")).suffix
    path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            path = Path(tmp.name)
        yield path
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


def sha256_file(path: str | Path) -> str | None:
    """Return a model artifact hash, or ``None`` for hub names/non-files."""

    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_detector(detector: Any, requested_model: str) -> dict[str, Any]:
    """Describe the active detector without relying on a specific YOLO version."""

    model = getattr(detector, "model", None)
    raw_names = getattr(model, "names", {})
    if isinstance(raw_names, dict):
        classes = [str(raw_names[key]).lower() for key in sorted(raw_names)]
    elif isinstance(raw_names, (list, tuple)):
        classes = [str(value).lower() for value in raw_names]
    else:
        classes = []
    has_person = "person" in classes
    ppe_classes = sorted(set(classes) & PPE_CLASS_NAMES)
    capabilities: list[str] = []
    if has_person:
        capabilities.extend(["person_detection", "zone_intrusion"])
    if ppe_classes:
        capabilities.extend(["ppe_detection", "ppe_rule_events"])
    return {
        # A basename plus the checkpoint hash is enough for the Web audit view
        # and does not publish a user's home or mounted-volume path.
        "requested_model": Path(requested_model).name,
        "device": str(getattr(detector, "device", "unknown")),
        "sha256": sha256_file(requested_model),
        "classes": classes,
        "ppe_classes": ppe_classes,
        "capabilities": capabilities,
    }


def resolve_run_artifact(run_dir: str | Path, stored_path: str | Path) -> Path:
    """Resolve a portable run-relative artifact path for local presentation."""

    root = Path(run_dir).resolve()
    path = Path(stored_path)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Run artifact path escapes its run directory")
    return resolved


def _parse_required_ppe(raw: Any, zone_id: str) -> tuple[str, ...]:
    if not isinstance(raw, list):
        raise ZoneValidationError(f"Zone {zone_id!r} required_ppe must be a list.")
    if any(not isinstance(item, str) for item in raw):
        raise ZoneValidationError(
            f"Zone {zone_id!r} required_ppe values must be strings."
        )
    if any(not item or item.strip() != item for item in raw):
        raise ZoneValidationError(
            f"Zone {zone_id!r} required_ppe values must be non-empty canonical names."
        )
    if len(set(raw)) != len(raw):
        raise ZoneValidationError(
            f"Zone {zone_id!r} required_ppe values must not contain duplicates."
        )
    invalid = set(raw) - PPE_TYPES
    if invalid:
        raise ZoneValidationError(
            f"Zone {zone_id!r} has unsupported required PPE: {sorted(invalid)}."
        )
    return tuple(raw)


def _parse_dwell_seconds(raw: Any, zone_id: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ZoneValidationError(f"Zone {zone_id!r} dwell_seconds must be numeric.")
    value = float(raw)
    if not math.isfinite(value) or value < 0:
        raise ZoneValidationError(
            f"Zone {zone_id!r} dwell_seconds must be a finite non-negative value."
        )
    return value


def _parse_polygon(raw: Any, zone_id: str) -> tuple[tuple[float, float], ...]:
    if not isinstance(raw, list) or len(raw) < 3:
        raise ZoneValidationError(f"Zone {zone_id!r} polygon requires at least 3 points.")
    points: list[tuple[float, float]] = []
    for point_index, point in enumerate(raw, start=1):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ZoneValidationError(
                f"Zone {zone_id!r} point #{point_index} must contain exactly x and y."
            )
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError) as exc:
            raise ZoneValidationError(
                f"Zone {zone_id!r} point #{point_index} must be numeric."
            ) from exc
        if not math.isfinite(x) or not math.isfinite(y):
            raise ZoneValidationError(
                f"Zone {zone_id!r} point #{point_index} must be finite."
            )
        points.append((x, y))
    return tuple(points)


def _looks_normalized(polygon: tuple[tuple[float, float], ...]) -> bool:
    return all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in polygon)


def _validate_normalized_polygon(
    polygon: tuple[tuple[float, float], ...], zone_id: str
) -> None:
    if not _looks_normalized(polygon):
        raise ZoneValidationError(
            f"Zone {zone_id!r} normalized coordinates must all be between 0 and 1."
        )
    unique_points = set(polygon)
    if len(unique_points) < 3:
        raise ZoneValidationError(f"Zone {zone_id!r} requires at least 3 unique points.")
    area = 0.0
    for index, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(index + 1) % len(polygon)]
        area += x1 * y2 - x2 * y1
    if abs(area) < 1e-9:
        raise ZoneValidationError(f"Zone {zone_id!r} polygon area must be greater than zero.")
