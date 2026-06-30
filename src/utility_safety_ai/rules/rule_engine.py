"""Convert YOLO detections and zone geometry into risk-ranked safety events."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from ..compliance.person_ppe_association import (
    NEGATIVE_PPE,
    associate_ppe_to_persons,
)
from ..events.event import Detection, SafetyEvent
from ..i18n import _
from ..zones.geometry import bottom_center, point_in_polygon
from ..zones.zone import Zone
from . import risk

logger = logging.getLogger(__name__)

# Maps a negative PPE class name to (event_type, description_i18n_key, default_risk).
PPE_RULES: dict[str, tuple[str, str, str]] = {
    "no_helmet": ("missing_helmet", "missing_helmet", risk.MEDIUM),
    "no_vest": ("missing_vest", "missing_vest", risk.MEDIUM),
    "no_gloves": ("missing_gloves", "missing_gloves", risk.LOW),
    "no_boots": ("missing_boots", "missing_boots", risk.MEDIUM),
    "no_goggles": ("missing_goggles", "missing_goggles", risk.MEDIUM),
    "no_goggle": ("missing_goggles", "missing_goggles", risk.MEDIUM),
}

ZONE_EVENT_TYPE = "zone_intrusion"
CRITICAL_EVENT_TYPE = "critical_helmet_in_zone"
MULTIPLE_PPE_EVENT_TYPE = "multiple_ppe_violations"


class RuleEngine:
    """Evaluate safety rules with cooldown-based de-duplication."""

    def __init__(
        self,
        zones: list[Zone],
        cooldown_seconds: float = 10.0,
        rules_config: dict[str, Any] | None = None,
    ) -> None:
        self.zones = zones
        self.cooldown_seconds = cooldown_seconds
        self.rules_config = rules_config or {}
        self._last_emitted: dict[tuple[int | None, str, str | None], float] = {}

    def evaluate(
        self,
        detections: list[Detection],
        source_type: str,
        source_path: str,
        frame_index: int | None = None,
        time_seconds: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[SafetyEvent]:
        """Evaluate all rules for a single frame/image.

        Args:
            detections: List of normalized detections.
            source_type: ``image``, ``video``, or ``camera``.
            source_path: Path or identifier of the source.
            frame_index: Frame number for video/camera sources.
            time_seconds: Elapsed time for video/camera sources.
            metadata: Extra context to attach to every event.

        Returns:
            New safety events that pass the cooldown filter.
        """
        metadata = metadata or {}
        timestamp = datetime.now(timezone.utc).isoformat()

        compliance_records, unassociated = associate_ppe_to_persons(detections)
        events: list[SafetyEvent] = []

        for record in compliance_records:
            person_track_id = record.person_track_id

            # PPE violation events for this person (one per missing item).
            ppe_events: list[SafetyEvent] = []
            for neg_det in record.negative_detections:
                rule = PPE_RULES.get(neg_det.class_name)
                if rule is None:
                    continue
                event_type, description_key, default_risk = rule
                ppe_type = NEGATIVE_PPE.get(neg_det.class_name, neg_det.class_name)
                risk_level = self.rules_config.get("risk_levels", {}).get(
                    event_type, default_risk
                )
                ppe_events.append(
                    SafetyEvent(
                        event_id=str(uuid.uuid4()),
                        timestamp=timestamp,
                        source_type=source_type,
                        source_path=source_path,
                        frame_index=frame_index,
                        time_seconds=time_seconds,
                        risk_level=risk_level,
                        event_type=event_type,
                        description=f"{_(description_key)} (person #{person_track_id})",
                        person_track_id=person_track_id,
                        bbox=record.bbox,
                        zone_id=None,
                        zone_name=None,
                        snapshot_path=None,
                        metadata={
                            **metadata,
                            "confidence": neg_det.confidence,
                            "ppe_type": ppe_type,
                            "compliance": record.summary(),
                        },
                    )
                )

            # Zone intrusion events for this person.
            zone_events = self._build_zone_events(
                record,
                timestamp,
                source_type,
                source_path,
                frame_index,
                time_seconds,
                metadata,
            )

            # Escalate zone events to critical if missing helmet + inside zone.
            if any(e.event_type == "missing_helmet" for e in ppe_events) and zone_events:
                zone_events = [
                    self._escalate(
                        self._with_metadata(
                            zone_event, escalation_reason="missing_helmet_in_zone"
                        ),
                        risk.CRITICAL,
                    )
                    for zone_event in zone_events
                ]

            # Escalate all PPE events to high when multiple PPE violations occur.
            if len(ppe_events) >= 2:
                ppe_events = [self._escalate(e, risk.HIGH) for e in ppe_events]

            for event in ppe_events + zone_events:
                if self._accept(event, time_seconds):
                    events.append(event)

        # Handle negative PPE detections that could not be associated with a person.
        # These are still emitted as standalone violations so they are not lost,
        # but the metadata notes that no person was associated.
        for det in unassociated:
            rule = PPE_RULES.get(det.class_name)
            if rule is None:
                continue
            event_type, description_key, default_risk = rule
            risk_level = self.rules_config.get("risk_levels", {}).get(
                event_type, default_risk
            )
            event = SafetyEvent(
                event_id=str(uuid.uuid4()),
                timestamp=timestamp,
                source_type=source_type,
                source_path=source_path,
                frame_index=frame_index,
                time_seconds=time_seconds,
                risk_level=risk_level,
                event_type=event_type,
                description=f"{_(description_key)} (unassociated detection)",
                person_track_id=None,
                bbox=det.bbox,
                zone_id=None,
                zone_name=None,
                snapshot_path=None,
                metadata={
                    **metadata,
                    "confidence": det.confidence,
                    "associated": False,
                },
            )
            if self._accept(event, time_seconds):
                events.append(event)

        return events

    def _build_zone_events(
        self,
        record: Any,
        timestamp: str,
        source_type: str,
        source_path: str,
        frame_index: int | None,
        time_seconds: float | None,
        metadata: dict[str, Any],
    ) -> list[SafetyEvent]:
        """Build zone-intrusion events for a single person compliance record."""
        events: list[SafetyEvent] = []
        point = bottom_center(record.bbox)
        for zone in self.zones:
            if point_in_polygon(point, zone.polygon):
                risk_level = self.rules_config.get("risk_levels", {}).get(
                    ZONE_EVENT_TYPE, zone.risk_level
                )
                events.append(
                    SafetyEvent(
                        event_id=str(uuid.uuid4()),
                        timestamp=timestamp,
                        source_type=source_type,
                        source_path=source_path,
                        frame_index=frame_index,
                        time_seconds=time_seconds,
                        risk_level=risk_level,
                        event_type=ZONE_EVENT_TYPE,
                        description=f"{_('zone_intrusion')} #{record.person_track_id}: {zone.name}",
                        person_track_id=record.person_track_id,
                        bbox=record.bbox,
                        zone_id=zone.id,
                        zone_name=zone.name,
                        snapshot_path=None,
                        metadata={
                            **metadata,
                            "zone_risk": zone.risk_level,
                            "compliance": record.summary(),
                        },
                    )
                )
        return events

    def _with_metadata(self, event: SafetyEvent, **updates) -> SafetyEvent:
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
        if risk.RISK_ORDER.get(event.risk_level, -1) >= risk.RISK_ORDER.get(new_risk, -1):
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
        """Apply cooldown de-duplication.

        Events are keyed by (track_id, event_type, zone_id). For video/camera,
        repeated events are suppressed until the cooldown window expires. For
        single-frame images there is no temporal duplication, so every distinct
        event is accepted.
        """
        key = (event.person_track_id, event.event_type, event.zone_id)
        last = self._last_emitted.get(key)

        if time_seconds is None:
            # Single image: do not suppress events just because they share a
            # kind; track IDs are already stable per image.
            return True

        if last is None:
            self._last_emitted[key] = time_seconds
            return True

        if time_seconds - last >= self.cooldown_seconds:
            self._last_emitted[key] = time_seconds
            return True

        return False
