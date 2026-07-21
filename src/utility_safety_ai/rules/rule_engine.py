"""Convert resolved PPE and zone state into risk-ranked safety events."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Hashable
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from typing import Any

from ..compliance.person_ppe_association import (
    NEGATIVE_PPE,
    PPE_TYPES,
    PersonCompliance,
    associate_ppe_to_persons,
)
from ..events.event import Detection, SafetyEvent
from ..i18n import _
from ..zones.geometry import bottom_center
from ..zones.zone import Zone
from . import risk

logger = logging.getLogger(__name__)

# Maps a normalized PPE type to (event_type, description_i18n_key, default_risk).
PPE_TYPE_RULES: dict[str, tuple[str, str, str]] = {
    "helmet": ("missing_helmet", "missing_helmet", risk.MEDIUM),
    "vest": ("missing_vest", "missing_vest", risk.MEDIUM),
    "gloves": ("missing_gloves", "missing_gloves", risk.LOW),
    "boots": ("missing_boots", "missing_boots", risk.MEDIUM),
    "goggles": ("missing_goggles", "missing_goggles", risk.MEDIUM),
}

# Backwards-compatible class-name lookup used by integrations and older configs.
PPE_RULES: dict[str, tuple[str, str, str]] = {
    class_name: PPE_TYPE_RULES[ppe_type]
    for class_name, ppe_type in NEGATIVE_PPE.items()
}

ZONE_EVENT_TYPE = "zone_intrusion"
CRITICAL_EVENT_TYPE = "critical_helmet_in_zone"
MULTIPLE_PPE_EVENT_TYPE = "multiple_ppe_violations"


@dataclass(frozen=True)
class RuleEvaluation:
    """Result for one frame, separating current truth from log emissions.

    ``active_findings`` contains every rule that is true in the current frame.
    ``new_events`` is the cooldown-filtered subset suitable for snapshots and
    persistent event logs.
    """

    active_findings: list[SafetyEvent]
    confirmed_findings: list[SafetyEvent]
    provisional_findings: list[SafetyEvent]
    new_events: list[SafetyEvent]
    resolved_findings: list[SafetyEvent]


class RuleEngine:
    """Evaluate safety rules with cooldown and zone dwell state."""

    def __init__(
        self,
        zones: list[Zone],
        cooldown_seconds: float = 10.0,
        rules_config: dict[str, Any] | None = None,
    ) -> None:
        if (
            isinstance(cooldown_seconds, bool)
            or not isinstance(cooldown_seconds, (int, float))
            or not isfinite(float(cooldown_seconds))
            or cooldown_seconds < 0
        ):
            raise ValueError("cooldown_seconds must be a finite non-negative value")
        self.zones = list(zones)
        self.cooldown_seconds = float(cooldown_seconds)
        self.rules_config = dict(rules_config or {})
        self._required_ppe = self._parse_required_ppe(self.rules_config)
        self.association_min_score = self._bounded_config_float(
            "association_min_score", 0.28
        )
        self.association_ambiguity_margin = self._bounded_config_float(
            "association_ambiguity_margin", 0.08
        )
        self._validate_risk_overrides()
        self._last_emitted: dict[
            tuple[int | None, str, str | None], float
        ] = {}
        # A tracker can assign a new ID after a short occlusion. Keep a small
        # spatial event cache so an ID switch cannot bypass the operator's
        # cooldown and flood the event log with the same physical finding.
        self._recent_spatial_events: list[tuple[SafetyEvent, float]] = []
        self._active_findings: list[SafetyEvent] = []
        self._zone_entered_at: dict[tuple[Hashable, str], float] = {}
        self._stream_key: tuple[str, str] | None = None
        self._last_time_seconds: float | None = None

    def reset(self) -> None:
        """Reset all temporal state before processing a new run/stream."""
        self._last_emitted.clear()
        self._recent_spatial_events.clear()
        self._active_findings.clear()
        self._zone_entered_at.clear()
        self._stream_key = None
        self._last_time_seconds = None

    def evaluate(
        self,
        detections: list[Detection],
        source_type: str,
        source_path: str,
        frame_index: int | None = None,
        time_seconds: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[SafetyEvent]:
        """Return new cooldown-filtered events for compatibility.

        Call :meth:`evaluate_frame` when annotations need every currently active
        finding while logs should receive only newly emitted events.
        """
        return self.evaluate_frame(
            detections=detections,
            source_type=source_type,
            source_path=source_path,
            frame_index=frame_index,
            time_seconds=time_seconds,
            metadata=metadata,
        ).new_events

    def evaluate_frame(
        self,
        detections: list[Detection],
        source_type: str,
        source_path: str,
        frame_index: int | None = None,
        time_seconds: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RuleEvaluation:
        """Evaluate one image/frame and expose active plus newly emitted events."""
        if (
            time_seconds is not None
            and (
                isinstance(time_seconds, bool)
                or not isinstance(time_seconds, (int, float))
                or not isfinite(float(time_seconds))
                or time_seconds < 0
            )
        ):
            raise ValueError("time_seconds must be a finite non-negative value")
        self._prepare_stream(source_type, source_path, time_seconds)
        shared_metadata = dict(metadata or {})
        timestamp = datetime.now(timezone.utc).isoformat()
        frame_size = self._frame_size(shared_metadata)

        compliance_records, _unassociated = associate_ppe_to_persons(
            detections,
            iou_threshold=self.association_min_score,
            ambiguity_margin=self.association_ambiguity_margin,
        )
        active_findings: list[SafetyEvent] = []
        current_zone_keys: set[tuple[Hashable, str]] = set()

        for record_index, record in enumerate(compliance_records):
            person_key: Hashable = (
                record.person_track_id
                if record.person_track_id is not None
                else ("untracked", record_index, record.bbox)
            )
            zone_events, mature_zones = self._build_zone_events(
                record=record,
                person_key=person_key,
                current_zone_keys=current_zone_keys,
                timestamp=timestamp,
                source_type=source_type,
                source_path=source_path,
                frame_index=frame_index,
                time_seconds=time_seconds,
                metadata=shared_metadata,
                frame_size=frame_size,
            )

            required_ppe = set(self._required_ppe)
            for zone in mature_zones:
                required_ppe.update(zone.required_ppe)
            violation_types = [
                ppe_type
                for ppe_type in record.violations()
                if ppe_type in required_ppe
            ]
            ppe_events = [
                self._build_ppe_event(
                    record=record,
                    ppe_type=ppe_type,
                    timestamp=timestamp,
                    source_type=source_type,
                    source_path=source_path,
                    frame_index=frame_index,
                    time_seconds=time_seconds,
                    metadata=shared_metadata,
                )
                for ppe_type in violation_types
            ]

            # Escalation is based on resolved, distinct PPE types. Repeated boxes
            # for two hands or contradictory positive/negative classes therefore
            # cannot manufacture a multi-violation alarm.
            if len(violation_types) >= 2:
                ppe_events = [self._escalate(event, risk.HIGH) for event in ppe_events]

            if "helmet" in violation_types and zone_events:
                zone_events = [
                    self._escalate(
                        self._with_metadata(
                            event,
                            escalation_reason="missing_helmet_in_zone",
                        ),
                        risk.CRITICAL,
                    )
                    for event in zone_events
                ]

            active_findings.extend(ppe_events)
            active_findings.extend(zone_events)

        # Leaving a zone resets dwell accumulation for the next entry.
        self._zone_entered_at = {
            key: entered_at
            for key, entered_at in self._zone_entered_at.items()
            if key in current_zone_keys
        }

        active_findings, resolved_findings = self._apply_lifecycle(
            active_findings, timestamp
        )
        confirmed_findings = [
            event
            for event in active_findings
            if event.metadata.get("confirmation_status") == "confirmed"
        ]
        provisional_findings = [
            event
            for event in active_findings
            if event.metadata.get("confirmation_status") == "observing"
        ]
        new_events = [
            event
            for event in confirmed_findings
            if self._accept(event, time_seconds)
        ]
        return RuleEvaluation(
            active_findings=active_findings,
            confirmed_findings=confirmed_findings,
            provisional_findings=provisional_findings,
            new_events=new_events,
            resolved_findings=resolved_findings,
        )

    def _apply_lifecycle(
        self,
        current: list[SafetyEvent],
        timestamp: str,
    ) -> tuple[list[SafetyEvent], list[SafetyEvent]]:
        """Mark findings opened/ongoing/resolved without duplicating event logs."""

        unmatched_previous = list(self._active_findings)
        active: list[SafetyEvent] = []
        for finding in current:
            match_index = next(
                (
                    index
                    for index, previous in enumerate(unmatched_previous)
                    if previous.event_type == finding.event_type
                    and previous.zone_id == finding.zone_id
                    and self._same_physical_finding(previous, finding)
                ),
                None,
            )
            state = "opened"
            confirmation_count = 1
            if match_index is not None:
                previous = unmatched_previous.pop(match_index)
                state = "ongoing"
                confirmation_count = int(
                    previous.metadata.get("confirmation_count", 1)
                ) + 1
            confirmation_required = self._confirmation_required(finding)
            active.append(
                self._with_metadata(
                    finding,
                    lifecycle_state=state,
                    confirmation_count=confirmation_count,
                    confirmation_required=confirmation_required,
                    confirmation_status=(
                        "confirmed"
                        if confirmation_count >= confirmation_required
                        else "observing"
                    ),
                )
            )
        resolved = [
            self._with_metadata(
                previous,
                lifecycle_state="resolved",
                resolved_at=timestamp,
            )
            for previous in unmatched_previous
        ]
        self._active_findings = active
        return active, resolved

    def _confirmation_required(self, event: SafetyEvent) -> int:
        """Return consecutive observations required before event emission."""
        # A still image has no temporal evidence to accumulate. Its explicit
        # detections remain immediately reviewable for backwards compatibility.
        if event.source_type == "image":
            return 1
        configured = self.rules_config.get("confirmation_frames", 1)
        if isinstance(configured, dict):
            raw_value = configured.get(event.event_type, configured.get("default", 1))
        else:
            raw_value = configured
        if isinstance(raw_value, bool) or not isinstance(raw_value, int) or raw_value < 1:
            raise ValueError(
                "rules_config.confirmation_frames values must be positive integers"
            )
        return raw_value

    def _build_ppe_event(
        self,
        record: PersonCompliance,
        ppe_type: str,
        timestamp: str,
        source_type: str,
        source_path: str,
        frame_index: int | None,
        time_seconds: float | None,
        metadata: dict[str, Any],
    ) -> SafetyEvent:
        event_type, description_key, default_risk = PPE_TYPE_RULES[ppe_type]
        evidence = record.negative_evidence(ppe_type)
        event_metadata: dict[str, Any] = {
            **metadata,
            "ppe_type": ppe_type,
            "compliance": record.summary(),
        }
        if evidence is not None:
            event_metadata["confidence"] = evidence.confidence
            event_metadata["evidence_class"] = evidence.class_name
        return SafetyEvent(
            event_id=str(uuid.uuid4()),
            timestamp=timestamp,
            source_type=source_type,
            source_path=source_path,
            frame_index=frame_index,
            time_seconds=time_seconds,
            risk_level=self._risk_for(event_type, default_risk),
            event_type=event_type,
            description=f"{_(description_key)} (person #{record.person_track_id})",
            person_track_id=record.person_track_id,
            bbox=record.bbox,
            zone_id=None,
            zone_name=None,
            snapshot_path=None,
            metadata=event_metadata,
        )

    def _build_zone_events(
        self,
        record: PersonCompliance,
        person_key: Hashable,
        current_zone_keys: set[tuple[Hashable, str]],
        timestamp: str,
        source_type: str,
        source_path: str,
        frame_index: int | None,
        time_seconds: float | None,
        metadata: dict[str, Any],
        frame_size: tuple[float, float] | None,
    ) -> tuple[list[SafetyEvent], list[Zone]]:
        events: list[SafetyEvent] = []
        mature_zones: list[Zone] = []
        point = bottom_center(record.bbox)
        for zone in self.zones:
            if not zone.contains(point, frame_size):
                continue
            key = (person_key, zone.id)
            current_zone_keys.add(key)
            entered_at = self._zone_entered_at.setdefault(
                key, time_seconds if time_seconds is not None else 0.0
            )
            dwell_seconds = self._zone_dwell_seconds(zone)
            elapsed = (
                None if time_seconds is None else max(0.0, time_seconds - entered_at)
            )
            if elapsed is not None and elapsed < dwell_seconds:
                continue

            mature_zones.append(zone)
            events.append(
                SafetyEvent(
                    event_id=str(uuid.uuid4()),
                    timestamp=timestamp,
                    source_type=source_type,
                    source_path=source_path,
                    frame_index=frame_index,
                    time_seconds=time_seconds,
                    risk_level=self._risk_for(ZONE_EVENT_TYPE, zone.risk_level),
                    event_type=ZONE_EVENT_TYPE,
                    description=(
                        f"{_('zone_intrusion')} #{record.person_track_id}: {zone.name}"
                    ),
                    person_track_id=record.person_track_id,
                    bbox=record.bbox,
                    zone_id=zone.id,
                    zone_name=zone.name,
                    snapshot_path=None,
                    metadata={
                        **metadata,
                        "zone_risk": zone.risk_level,
                        "dwell_seconds": dwell_seconds,
                        "time_in_zone_seconds": elapsed,
                        "required_ppe": list(zone.required_ppe),
                        "compliance": record.summary(),
                    },
                )
            )
        return events, mature_zones

    def _prepare_stream(
        self,
        source_type: str,
        source_path: str,
        time_seconds: float | None,
    ) -> None:
        stream_key = (source_type, source_path)
        time_restarted = (
            time_seconds is not None
            and self._last_time_seconds is not None
            and time_seconds < self._last_time_seconds
        )
        if self._stream_key is not None and (
            stream_key != self._stream_key or time_restarted
        ):
            self.reset()
        self._stream_key = stream_key
        if time_seconds is not None:
            self._last_time_seconds = time_seconds

    def _risk_for(self, event_type: str, default: str) -> str:
        return self.rules_config.get("risk_levels", {}).get(event_type, default)

    def _zone_dwell_seconds(self, zone: Zone) -> float:
        overrides = self.rules_config.get("zone_dwell_seconds", {})
        if not isinstance(overrides, dict):
            raise ValueError("rules_config.zone_dwell_seconds must be a mapping")
        value = overrides.get(zone.id, zone.dwell_seconds)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            or value < 0
        ):
            raise ValueError(
                f"Dwell time for zone {zone.id!r} must be finite and non-negative"
            )
        return float(value)

    @staticmethod
    def _frame_size(metadata: dict[str, Any]) -> tuple[float, float] | None:
        raw_size = metadata.get("frame_size")
        if raw_size is None:
            width = metadata.get("frame_width", metadata.get("image_width"))
            height = metadata.get("frame_height", metadata.get("image_height"))
            if width is None and height is None:
                return None
            raw_size = (width, height)
        if (
            not isinstance(raw_size, (list, tuple))
            or len(raw_size) != 2
            or any(isinstance(value, bool) for value in raw_size)
        ):
            raise ValueError("metadata.frame_size must be (width, height)")
        try:
            width, height = (float(raw_size[0]), float(raw_size[1]))
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata.frame_size must be numeric") from exc
        if not isfinite(width) or not isfinite(height) or width <= 0 or height <= 0:
            raise ValueError("metadata.frame_size width and height must be positive")
        return (width, height)

    @staticmethod
    def _parse_required_ppe(config: dict[str, Any]) -> tuple[str, ...]:
        configured = config.get("required_ppe", PPE_TYPES)
        if isinstance(configured, str) or not isinstance(configured, (list, tuple)):
            raise ValueError("rules_config.required_ppe must be a list")
        required = tuple(dict.fromkeys(configured))
        invalid = set(required) - set(PPE_TYPES)
        if invalid:
            raise ValueError(f"Unsupported required PPE types: {sorted(invalid)}")
        return required

    def _validate_risk_overrides(self) -> None:
        overrides = self.rules_config.get("risk_levels", {})
        if not isinstance(overrides, dict):
            raise ValueError("rules_config.risk_levels must be a mapping")
        invalid = {
            event_type: level
            for event_type, level in overrides.items()
            if level not in risk.RISK_ORDER
        }
        if invalid:
            raise ValueError(f"Invalid risk-level overrides: {invalid}")

    def _bounded_config_float(self, name: str, default: float) -> float:
        value = self.rules_config.get(name, default)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(f"rules_config.{name} must be a finite value between 0 and 1")
        return float(value)

    def _with_metadata(self, event: SafetyEvent, **updates: Any) -> SafetyEvent:
        """Return a copy of the event with updated metadata."""
        return SafetyEvent(
            event_id=event.event_id,
            timestamp=event.timestamp,
            source_type=event.source_type,
            source_path=event.source_path,
            frame_index=event.frame_index,
            time_seconds=event.time_seconds,
            risk_level=event.risk_level,
            event_type=event.event_type,
            description=event.description,
            person_track_id=event.person_track_id,
            bbox=event.bbox,
            zone_id=event.zone_id,
            zone_name=event.zone_name,
            snapshot_path=event.snapshot_path,
            metadata={**event.metadata, **updates},
        )

    def _escalate(self, event: SafetyEvent, new_risk: str) -> SafetyEvent:
        """Return a new event with an escalated risk level."""
        if risk.RISK_ORDER.get(event.risk_level, -1) >= risk.RISK_ORDER.get(
            new_risk, -1
        ):
            return event
        return SafetyEvent(
            event_id=event.event_id,
            timestamp=event.timestamp,
            source_type=event.source_type,
            source_path=event.source_path,
            frame_index=event.frame_index,
            time_seconds=event.time_seconds,
            risk_level=new_risk,
            event_type=event.event_type,
            description=event.description,
            person_track_id=event.person_track_id,
            bbox=event.bbox,
            zone_id=event.zone_id,
            zone_name=event.zone_name,
            snapshot_path=event.snapshot_path,
            metadata={**event.metadata, "original_risk": event.risk_level},
        )

    def _accept(self, event: SafetyEvent, time_seconds: float | None) -> bool:
        """Apply cooldown de-duplication to a currently active finding."""
        key = (event.person_track_id, event.event_type, event.zone_id)
        last = self._last_emitted.get(key)

        if time_seconds is None:
            return True
        if last is not None and time_seconds - last < self.cooldown_seconds:
            return False

        cutoff = time_seconds - self.cooldown_seconds
        self._recent_spatial_events = [
            item for item in self._recent_spatial_events if item[1] >= cutoff
        ]
        if any(
            previous.event_type == event.event_type
            and previous.zone_id == event.zone_id
            and self._same_physical_finding(previous, event)
            for previous, _emitted_at in self._recent_spatial_events
        ):
            return False

        self._last_emitted[key] = time_seconds
        self._recent_spatial_events.append((event, time_seconds))
        return True

    @staticmethod
    def _same_physical_finding(previous: SafetyEvent, current: SafetyEvent) -> bool:
        """Return whether two events likely describe one person after an ID switch.

        Track IDs remain the strongest identity signal. When they differ, a
        conservative overlap check reconnects only substantially overlapping
        person boxes, avoiding the common short-occlusion alert storm without
        merging separate workers elsewhere in the frame.
        """

        if previous.person_track_id == current.person_track_id:
            return True
        if previous.bbox is None or current.bbox is None:
            return False
        ax1, ay1, ax2, ay2 = previous.bbox
        bx1, by1, bx2, by2 = current.bbox
        intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
            0.0, min(ay2, by2) - max(ay1, by1)
        )
        if intersection <= 0:
            return False
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        smaller_area = min(area_a, area_b)
        # Intersection-over-smaller-area handles a partially visible worker
        # whose replacement box is nested inside the pre-occlusion box.
        return smaller_area > 0 and intersection / smaller_area >= 0.45
