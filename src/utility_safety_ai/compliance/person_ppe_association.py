"""Associate PPE detections with person detections for compliance reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

from ..events.event import Detection

# Positive PPE classes and the body part / PPE type they represent.
POSITIVE_PPE = {
    "helmet": "helmet",
    "vest": "vest",
    "gloves": "gloves",
    "boots": "boots",
    "goggles": "goggles",
}

# Explicit negative classes and the corresponding PPE type they report.
NEGATIVE_PPE = {
    "no_helmet": "helmet",
    "no_vest": "vest",
    "no_gloves": "gloves",
    "no_boots": "boots",
    "no_goggles": "goggles",
    "no_goggle": "goggles",
}

PPE_TYPES = ["helmet", "vest", "gloves", "boots", "goggles"]

# Expected vertical location of PPE-box centres inside a person box. These broad
# bands intentionally tolerate posture and viewpoint variation while preventing
# a boot beside one worker from being assigned to the torso of another.
_BODY_REGIONS: dict[str, tuple[float, float]] = {
    "helmet": (-0.08, 0.34),
    "goggles": (-0.02, 0.38),
    "vest": (0.12, 0.72),
    "gloves": (0.20, 0.88),
    "boots": (0.62, 1.08),
}


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def _center_in_box(
    bbox: tuple[float, float, float, float],
    container: tuple[float, float, float, float],
) -> bool:
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    cx1, cy1, cx2, cy2 = container
    return cx1 <= cx <= cx2 and cy1 <= cy <= cy2


def _intersection_over_item(
    item: tuple[float, float, float, float],
    container: tuple[float, float, float, float],
) -> float:
    """Return the fraction of a small PPE box contained by a person box."""
    x1 = max(item[0], container[0])
    y1 = max(item[1], container[1])
    x2 = min(item[2], container[2])
    y2 = min(item[3], container[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    item_area = max(0.0, item[2] - item[0]) * max(0.0, item[3] - item[1])
    return intersection / item_area if item_area > 0 else 0.0


def _body_region_fit(
    ppe_bbox: tuple[float, float, float, float],
    person_bbox: tuple[float, float, float, float],
    ppe_type: str,
) -> float:
    height = person_bbox[3] - person_bbox[1]
    if height <= 0:
        return 0.0
    centre_y = (ppe_bbox[1] + ppe_bbox[3]) / 2.0
    relative_y = (centre_y - person_bbox[1]) / height
    lower, upper = _BODY_REGIONS[ppe_type]
    if lower <= relative_y <= upper:
        return 1.0
    distance = lower - relative_y if relative_y < lower else relative_y - upper
    return max(0.0, 1.0 - distance / 0.45)


def _horizontal_fit(
    ppe_bbox: tuple[float, float, float, float],
    person_bbox: tuple[float, float, float, float],
) -> float:
    width = person_bbox[2] - person_bbox[0]
    if width <= 0:
        return 0.0
    ppe_centre = (ppe_bbox[0] + ppe_bbox[2]) / 2.0
    person_centre = (person_bbox[0] + person_bbox[2]) / 2.0
    return max(0.0, 1.0 - abs(ppe_centre - person_centre) / (width * 0.75))


def _association_score(
    ppe_bbox: tuple[float, float, float, float],
    person_bbox: tuple[float, float, float, float],
    ppe_type: str,
) -> float:
    """Score containment, body-region plausibility, and horizontal alignment."""
    containment = _intersection_over_item(ppe_bbox, person_bbox)
    body_fit = _body_region_fit(ppe_bbox, person_bbox, ppe_type)
    horizontal_fit = _horizontal_fit(ppe_bbox, person_bbox)
    centre_bonus = 1.0 if _center_in_box(ppe_bbox, person_bbox) else 0.0
    # IoU is useful for large explicit-negative boxes, while containment is far
    # more informative for small helmets, gloves, and boots.
    return (
        0.40 * containment
        + 0.25 * body_fit
        + 0.18 * horizontal_fit
        + 0.12 * centre_bonus
        + 0.05 * _iou(ppe_bbox, person_bbox)
    )


@dataclass
class PersonCompliance:
    """Per-person PPE compliance summary."""

    person_track_id: int | None
    bbox: tuple[float, float, float, float]
    items: dict[str, str] = field(default_factory=dict)
    positive_detections: list[Detection] = field(default_factory=list)
    negative_detections: list[Detection] = field(default_factory=list)
    conflicting_detections: list[Detection] = field(default_factory=list)
    association_scores: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Ensure all PPE types are represented.
        for ppe_type in PPE_TYPES:
            self.items.setdefault(ppe_type, "unknown")

    def summary(self) -> dict[str, str]:
        """Return a readable compliance summary."""
        return {ppe_type: self.items.get(ppe_type, "unknown") for ppe_type in PPE_TYPES}

    def violations(self) -> list[str]:
        """Return distinct PPE types whose resolved state is missing."""
        return [ppe for ppe in PPE_TYPES if self.items.get(ppe) == "no"]

    def negative_evidence(self, ppe_type: str) -> Detection | None:
        """Return the selected negative evidence for a resolved PPE violation."""
        return next(
            (
                detection
                for detection in self.negative_detections
                if NEGATIVE_PPE.get(detection.class_name) == ppe_type
            ),
            None,
        )


def associate_ppe_to_persons(
    detections: list[Detection],
    iou_threshold: float = 0.28,
    *,
    ambiguity_margin: float = 0.08,
) -> tuple[list[PersonCompliance], list[Detection]]:
    """Associate PPE detections to person detections.

    Args:
        detections: All detections from a frame.
        iou_threshold: Minimum composite association score for a link.
        ambiguity_margin: Reject an association when the top two people have
            nearly identical scores, which is safer in crowded scenes.

    Returns:
        A tuple of (person compliance records, unassociated PPE detections).
    """
    if (
        isinstance(iou_threshold, bool)
        or not isinstance(iou_threshold, (int, float))
        or not isfinite(float(iou_threshold))
        or not 0.0 <= iou_threshold <= 1.0
    ):
        raise ValueError("iou_threshold must be a finite value between 0 and 1")
    if (
        isinstance(ambiguity_margin, bool)
        or not isinstance(ambiguity_margin, (int, float))
        or not isfinite(float(ambiguity_margin))
        or not 0.0 <= ambiguity_margin <= 1.0
    ):
        raise ValueError("ambiguity_margin must be a finite value between 0 and 1")
    persons = [d for d in detections if d.class_name == "person"]
    ppe_dets = [
        d
        for d in detections
        if d.class_name in POSITIVE_PPE or d.class_name in NEGATIVE_PPE
    ]

    # Index by detection position rather than track ID. Multiple untracked people
    # legitimately have ``track_id=None`` and must not overwrite one another.
    records = [
        PersonCompliance(person_track_id=person.track_id, bbox=person.bbox)
        for person in persons
    ]
    candidates: list[dict[str, dict[str, list[Detection]]]] = [
        {
            ppe_type: {"positive": [], "negative": []}
            for ppe_type in PPE_TYPES
        }
        for _ in persons
    ]

    unassociated: list[Detection] = []

    for ppe in ppe_dets:
        ppe_type = POSITIVE_PPE.get(ppe.class_name) or NEGATIVE_PPE.get(ppe.class_name)
        if ppe_type is None:
            unassociated.append(ppe)
            continue
        ranked: list[tuple[float, int]] = []
        for person_index, person in enumerate(persons):
            ranked.append(
                (_association_score(ppe.bbox, person.bbox, ppe_type), person_index)
            )
        ranked.sort(reverse=True)

        if not ranked or ranked[0][0] < iou_threshold:
            unassociated.append(ppe)
            continue
        if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < ambiguity_margin:
            unassociated.append(ppe)
            continue

        best_score, best_person_index = ranked[0]
        records[best_person_index].association_scores[ppe_type] = max(
            records[best_person_index].association_scores.get(ppe_type, 0.0),
            round(best_score, 4),
        )

        if ppe.class_name in POSITIVE_PPE:
            candidates[best_person_index][ppe_type]["positive"].append(ppe)
        else:
            candidates[best_person_index][ppe_type]["negative"].append(ppe)

    # Resolve every person/PPE pair once, independently of detector output order.
    # Positive evidence is conservative: it prevents a contradictory negative
    # classifier box from producing a false safety event. Repeated detections for
    # the same PPE type are represented by the highest-confidence box only.
    for record, by_type in zip(records, candidates, strict=True):
        for ppe_type in PPE_TYPES:
            positive = sorted(
                by_type[ppe_type]["positive"],
                key=lambda detection: detection.confidence,
                reverse=True,
            )
            negative = sorted(
                by_type[ppe_type]["negative"],
                key=lambda detection: detection.confidence,
                reverse=True,
            )
            if positive:
                record.items[ppe_type] = "yes"
                record.positive_detections.append(positive[0])
                record.conflicting_detections.extend(negative)
            elif negative:
                record.items[ppe_type] = "no"
                record.negative_detections.append(negative[0])

    return records, unassociated
