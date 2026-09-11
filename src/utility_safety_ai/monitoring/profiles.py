"""Opinionated monitoring profiles for common operating conditions."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MonitoringProfile:
    """A named bundle of detector, association, tracking, and alert settings."""

    name: str
    label: str
    description: str
    confidence: float
    nms_iou: float
    ppe_confirmation_frames: int
    zone_confirmation_frames: int
    association_min_score: float
    association_ambiguity_margin: float
    tracker_iou_threshold: float
    tracker_max_age: int

    def rule_config(self) -> dict[str, object]:
        """Return the RuleEngine fragment controlled by this profile."""
        return {
            "association_min_score": self.association_min_score,
            "association_ambiguity_margin": self.association_ambiguity_margin,
            "confirmation_frames": {
                "default": self.ppe_confirmation_frames,
                "zone_intrusion": self.zone_confirmation_frames,
            },
        }

    def manifest(self) -> dict[str, object]:
        """Return a JSON-safe representation for run audit manifests."""
        return asdict(self)


_PROFILES: dict[str, MonitoringProfile] = {
    "balanced": MonitoringProfile(
        name="balanced",
        label="Balanced",
        description="Stable alerts for everyday review with moderate sensitivity.",
        confidence=0.25,
        nms_iou=0.45,
        ppe_confirmation_frames=3,
        zone_confirmation_frames=2,
        association_min_score=0.28,
        association_ambiguity_margin=0.08,
        tracker_iou_threshold=0.22,
        tracker_max_age=12,
    ),
    "high_precision": MonitoringProfile(
        name="high_precision",
        label="High precision",
        description="Fewer review interruptions; requires stronger and longer evidence.",
        confidence=0.35,
        nms_iou=0.40,
        ppe_confirmation_frames=5,
        zone_confirmation_frames=3,
        association_min_score=0.36,
        association_ambiguity_margin=0.12,
        tracker_iou_threshold=0.28,
        tracker_max_age=10,
    ),
    "high_sensitivity": MonitoringProfile(
        name="high_sensitivity",
        label="High sensitivity",
        description="Surfaces weaker evidence sooner for high-recall investigation.",
        confidence=0.15,
        nms_iou=0.50,
        ppe_confirmation_frames=2,
        zone_confirmation_frames=1,
        association_min_score=0.20,
        association_ambiguity_margin=0.04,
        tracker_iou_threshold=0.16,
        tracker_max_age=16,
    ),
}


def get_monitoring_profile(name: str = "balanced") -> MonitoringProfile:
    """Return a validated monitoring profile by name."""
    try:
        return _PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(_PROFILES))
        raise ValueError(f"Unknown monitoring profile {name!r}; choose one of: {choices}") from exc


def monitoring_profile_names() -> tuple[str, ...]:
    """Return profile names in the intended UI order."""
    return tuple(_PROFILES)
