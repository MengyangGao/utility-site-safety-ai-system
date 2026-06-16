"""Shared dataclasses for detections and safety events."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Detection:
    """Normalized detection produced by any object detector."""

    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]
    track_id: int | None = None


@dataclass(frozen=True)
class SafetyEvent:
    """A single safety event emitted by the rule engine."""

    event_id: str
    timestamp: str
    source_type: str
    source_path: str
    frame_index: int | None
    time_seconds: float | None
    risk_level: str
    event_type: str
    description: str
    person_track_id: int | None
    bbox: tuple[float, float, float, float] | None
    zone_id: str | None
    zone_name: str | None
    snapshot_path: str | None
    metadata: dict[str, Any] = field(default_factory=dict)
