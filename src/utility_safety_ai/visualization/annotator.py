"""Draw clean, professional safety annotations on frames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..events.event import Detection, SafetyEvent
from ..i18n import _, class_label, current_language, risk_label
from ..zones.zone import Zone

RISK_COLORS = {
    "low": (0, 200, 0),
    "medium": (0, 200, 255),
    "high": (0, 100, 255),
    "critical": (0, 0, 230),
}

PERSON_COLOR = (255, 140, 0)
POSITIVE_PPE_COLORS = {
    "helmet": (0, 180, 0),
    "vest": (0, 180, 0),
    "gloves": (0, 180, 0),
    "boots": (0, 180, 0),
    "goggles": (0, 180, 0),
}
NEGATIVE_PPE_COLORS = {
    "no_helmet": (0, 0, 230),
    "no_vest": (0, 0, 230),
    "no_gloves": (0, 0, 230),
    "no_boots": (0, 0, 230),
    "no_goggles": (0, 0, 230),
    "no_goggle": (0, 0, 230),
}
ANNOTATED_CLASSES = {"person", *POSITIVE_PPE_COLORS, *NEGATIVE_PPE_COLORS}

# Fonts that commonly cover Latin and CJK glyphs on different platforms.
_FONT_CANDIDATES = [
    # macOS / iOS
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
    # Windows
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/arial.ttf",
    # Linux
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]

_PIL_FONT_CACHE: dict[int, ImageFont.FreeTypeFont | None] = {}


@dataclass(frozen=True)
class _PILTextOperation:
    text: str
    x: int
    baseline_y: int
    color: tuple[int, int, int]
    font: ImageFont.FreeTypeFont


class _TextRenderer:
    """Batch PIL text so a frame crosses the BGR/RGB boundary only once."""

    def __init__(self) -> None:
        self._pil_operations: list[_PILTextOperation] = []

    def draw(
        self,
        image: np.ndarray,
        text: str,
        x: int,
        y: int,
        color: tuple[int, int, int],
        font_scale: float,
    ) -> None:
        if _needs_pil(text):
            font = _load_pil_font(_pil_font_size(font_scale))
            if font is not None:
                self._pil_operations.append(
                    _PILTextOperation(text, x, y, color, font)
                )
                return
        cv2.putText(
            image,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            1,
        )

    def flush(self, image: np.ndarray) -> None:
        """Render every queued PIL label with one conversion in each direction."""
        if not self._pil_operations:
            return
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        for operation in self._pil_operations:
            _, _, _, bottom = operation.font.getbbox(operation.text)
            draw.text(
                (operation.x, operation.baseline_y - bottom),
                operation.text,
                font=operation.font,
                fill=_bgr_to_rgb(operation.color),
            )
        image[:] = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
        self._pil_operations.clear()


def _load_pil_font(size: int) -> ImageFont.FreeTypeFont | None:
    """Load the first available system font at the requested pixel size."""
    if size not in _PIL_FONT_CACHE:
        font: ImageFont.FreeTypeFont | None = None
        for candidate in _FONT_CANDIDATES:
            path = Path(candidate)
            if path.exists():
                try:
                    font = ImageFont.truetype(str(path), size)
                    break
                except Exception:  # pragma: no cover - font may be unloadable
                    continue
        _PIL_FONT_CACHE[size] = font
    return _PIL_FONT_CACHE[size]


def _needs_pil(text: str) -> bool:
    """Use PIL rendering when the active language or text contains CJK glyphs."""
    if current_language() in {"zh-hans", "zh-hant"}:
        return True
    return any(ord(ch) > 127 for ch in text)


def _pil_font_size(font_scale: float) -> int:
    """Map OpenCV font_scale to a roughly-equivalent PIL point size."""
    return max(8, int(font_scale * 34))


def _text_size(text: str, font_scale: float) -> tuple[int, int]:
    """Return (width, height) for ``text`` using the current renderer."""
    if _needs_pil(text):
        font = _load_pil_font(_pil_font_size(font_scale))
        if font is not None:
            left, top, right, bottom = font.getbbox(text)
            return int(right - left), int(bottom - top)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    return int(tw), int(th)


def _bgr_to_rgb(color: tuple[int, int, int]) -> tuple[int, int, int]:
    return (color[2], color[1], color[0])


def _draw_text(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    font_scale: float = 0.5,
    text_renderer: _TextRenderer | None = None,
) -> None:
    """Draw text with the renderer that supports the current language."""
    renderer = text_renderer or _TextRenderer()
    renderer.draw(image, text, x, y, color, font_scale)
    if text_renderer is None:
        renderer.flush(image)


def annotate_image(
    image: np.ndarray,
    zones: list[Zone],
    detections: list[Detection],
    events: list[SafetyEvent],
) -> np.ndarray:
    """Draw zones, detections, and an event summary on a copy of the image."""
    canvas = image.copy()
    text_renderer = _TextRenderer()
    draw_zones(canvas, zones, text_renderer=text_renderer)
    draw_detections(canvas, detections, text_renderer=text_renderer)
    draw_event_summary(canvas, events, text_renderer=text_renderer)
    text_renderer.flush(canvas)
    return canvas


def draw_zones(
    image: np.ndarray,
    zones: list[Zone],
    *,
    text_renderer: _TextRenderer | None = None,
) -> None:
    """Render subtle restricted-zone overlays with readable labels."""
    if not zones:
        return
    renderer = text_renderer or _TextRenderer()
    overlay = image.copy()
    height, width = image.shape[:2]
    for zone in zones:
        polygon = zone.resolved_polygon((width, height))
        pts = np.array([[int(x), int(y)] for x, y in polygon], np.int32)
        if pts.size == 0:
            continue
        pts = pts.reshape((-1, 1, 2))
        color = RISK_COLORS.get(zone.risk_level, (128, 128, 128))
        cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)
        cv2.fillPoly(overlay, [pts], color)
        label = f"{zone.name} ({risk_label(zone.risk_level)})"
        tx, ty = int(pts[0][0][0]), int(pts[0][0][1]) - 6
        _draw_zone_label(
            image,
            label,
            tx,
            ty,
            color,
            font_scale=0.45,
            text_renderer=renderer,
        )
    # Blend overlay for a subtle fill.
    cv2.addWeighted(overlay, 0.18, image, 0.82, 0, image)
    if text_renderer is None:
        renderer.flush(image)


def draw_detections(
    image: np.ndarray,
    detections: list[Detection],
    *,
    text_renderer: _TextRenderer | None = None,
) -> None:
    """Render bounding boxes and class labels with simple label de-cluttering."""
    renderer = text_renderer or _TextRenderer()
    occupied: list[tuple[int, int, int, int]] = []
    for det in detections:
        # General COCO checkpoints may also return benches, vehicles, bags,
        # etc. Keep those records in the detection audit log, but do not let
        # unrelated classes clutter the safety review overlay.
        if det.class_name not in ANNOTATED_CLASSES:
            continue
        x1, y1, x2, y2 = map(int, det.bbox)
        color = _detection_color(det.class_name)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        label = f"{class_label(det.class_name)} {det.confidence:.2f}"
        if det.track_id is not None:
            label += f" #{det.track_id}"

        tw, th = _text_size(label, 0.45)
        tx = max(4, x1)
        # Start above the box; if there is not enough room, place inside the box.
        ty = max(th + 6, y1 - 4)
        if ty - th - 5 < 0:
            ty = y1 + th + 6

        # Avoid overlapping existing labels by shifting down.
        label_rect = (tx, ty - th - 5, tx + tw + 4, ty + 2)
        attempts = 0
        while any(_rects_intersect(label_rect, occ) for occ in occupied) and attempts < 10:
            ty += 16
            label_rect = (tx, ty - th - 5, tx + tw + 4, ty + 2)
            attempts += 1

        occupied.append(label_rect)
        _draw_label(
            image,
            label,
            tx,
            ty,
            color,
            font_scale=0.45,
            text_renderer=renderer,
        )

        if det.class_name == "person":
            bx, by = int((x1 + x2) / 2), y2
            cv2.circle(image, (bx, by), 4, (0, 0, 255), -1)
    if text_renderer is None:
        renderer.flush(image)


def _rects_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def draw_event_summary(
    image: np.ndarray,
    events: list[SafetyEvent],
    *,
    text_renderer: _TextRenderer | None = None,
) -> None:
    """Draw a compact top-right risk summary panel with high contrast."""
    if not events:
        return

    risk_counts: dict[str, int] = {}
    for event in events:
        risk_counts[event.risk_level] = risk_counts.get(event.risk_level, 0) + 1

    lines = []
    for level in ("critical", "high", "medium", "low"):
        count = risk_counts.get(level, 0)
        if count:
            lines.append((level, f"{risk_label(level)}: {count}"))

    if not lines:
        return

    margin = 12
    line_height = 22
    title_height = 24
    title_text = _("ui.events_title")
    title_tw, title_th = _text_size(title_text, 0.55)
    max_line_tw = max(_text_size(text, 0.55)[0] for _, text in lines)
    panel_w = max(240, max(title_tw, max_line_tw) + margin * 2 + 20)
    panel_h = title_height + len(lines) * line_height + margin

    h, w = image.shape[:2]
    # Place panel in the top-right corner to avoid detection labels that cluster
    # at the top-left of bounding boxes.
    x1 = max(margin, w - panel_w - margin)
    y1 = margin
    x2 = min(w, x1 + panel_w)
    y2 = min(h, y1 + panel_h)

    # Blend only the small panel ROI instead of copying/converting a full frame.
    panel = image[y1 : y2 + 1, x1 : x2 + 1]
    dark_background = np.full_like(panel, 20)
    cv2.addWeighted(dark_background, 0.90, panel, 0.10, 0, panel)
    cv2.rectangle(image, (x1, y1), (x2, y2), (180, 180, 180), 1)

    renderer = text_renderer or _TextRenderer()
    # Title.
    _draw_text(
        image,
        title_text,
        x1 + margin,
        y1 + title_height - 6,
        (255, 255, 255),
        font_scale=0.55,
        text_renderer=renderer,
    )

    y = y1 + title_height + line_height - 4
    for level, text in lines:
        color = RISK_COLORS.get(level, (255, 255, 255))
        # Draw a small colored indicator square.
        cv2.rectangle(image, (x1 + margin, y - 12), (x1 + margin + 10, y), color, -1)
        _draw_text(
            image,
            text,
            x1 + margin + 16,
            y,
            (255, 255, 255),
            font_scale=0.55,
            text_renderer=renderer,
        )
        y += line_height
    if text_renderer is None:
        renderer.flush(image)


def _detection_color(class_name: str) -> tuple[int, int, int]:
    if class_name == "person":
        return PERSON_COLOR
    if class_name in POSITIVE_PPE_COLORS:
        return POSITIVE_PPE_COLORS[class_name]
    if class_name in NEGATIVE_PPE_COLORS:
        return NEGATIVE_PPE_COLORS[class_name]
    return (200, 200, 200)


def _draw_label(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    font_scale: float = 0.5,
    text_renderer: _TextRenderer | None = None,
) -> None:
    """Draw text with a small contrasting background."""
    tw, th = _text_size(text, font_scale)
    tx = max(4, x)
    ty = max(th + 6, y)
    cv2.rectangle(image, (tx, ty - th - 5), (tx + tw + 4, ty + 2), color, -1)
    _draw_text(
        image,
        text,
        tx + 2,
        ty,
        (255, 255, 255),
        font_scale,
        text_renderer,
    )


def _draw_zone_label(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    font_scale: float = 0.5,
    text_renderer: _TextRenderer | None = None,
) -> None:
    """Draw a zone label with a dark background for readability over bright fills."""
    tw, th = _text_size(text, font_scale)
    tx = max(4, x)
    ty = max(th + 6, y)
    # Dark background with a colored left strip.
    cv2.rectangle(image, (tx, ty - th - 5), (tx + tw + 6, ty + 2), (20, 20, 20), -1)
    cv2.rectangle(image, (tx, ty - th - 5), (tx + 4, ty + 2), color, -1)
    _draw_text(
        image,
        text,
        tx + 6,
        ty,
        (255, 255, 255),
        font_scale,
        text_renderer,
    )
