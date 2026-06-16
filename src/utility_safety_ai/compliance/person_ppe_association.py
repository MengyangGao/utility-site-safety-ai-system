"""Associate PPE detections with person detections for compliance reporting."""

from __future__ import annotations

from dataclasses import dataclass, field

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


def _association_score(
    ppe_bbox: tuple[float, float, float, float],
    person_bbox: tuple[float, float, float, float],
) -> float:
    """Return a geometry-based association score between a PPE box and a person box.

    The score is the IoU, but a PPE box whose center lies inside the person box
    gets a minimum score of 0.05 so small/occluded PPE items are still associated.
    """
    score = _iou(ppe_bbox, person_bbox)
    if _center_in_box(ppe_bbox, person_bbox):
        score = max(score, 0.05)
    return score


@dataclass
class PersonCompliance:
    """Per-person PPE compliance summary."""

    person_track_id: int
    bbox: tuple[float, float, float, float]
    items: dict[str, str] = field(default_factory=dict)
    positive_detections: list[Detection] = field(default_factory=list)
    negative_detections: list[Detection] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Ensure all PPE types are represented.
        for ppe_type in PPE_TYPES:
            self.items.setdefault(ppe_type, "unknown")

    def summary(self) -> dict[str, str]:
        """Return a readable compliance summary."""
        return {ppe_type: self.items.get(ppe_type, "unknown") for ppe_type in PPE_TYPES}

    def violations(self) -> list[str]:
        """Return PPE types marked as missing."""
        return [ppe for ppe, status in self.items.items() if status == "no"]


def associate_ppe_to_persons(
    detections: list[Detection],
    iou_threshold: float = 0.05,
) -> tuple[list[PersonCompliance], list[Detection]]:
    """Associate PPE detections to person detections.

    Args:
        detections: All detections from a frame.
        iou_threshold: Minimum association score for a PPE box to be linked to a person.

    Returns:
        A tuple of (person compliance records, unassociated PPE detections).
    """
    persons = [d for d in detections if d.class_name == "person"]
    ppe_dets = [
        d
        for d in detections
        if d.class_name in POSITIVE_PPE or d.class_name in NEGATIVE_PPE
    ]

    compliance_by_person: dict[int, PersonCompliance] = {}
    for person in persons:
        compliance_by_person[person.track_id] = PersonCompliance(
            person_track_id=person.track_id,
            bbox=person.bbox,
        )

    unassociated: list[Detection] = []

    for ppe in ppe_dets:
        best_score = iou_threshold
        best_person: Detection | None = None
        for person in persons:
            score = _association_score(ppe.bbox, person.bbox)
            if score >= best_score:
                best_score = score
                best_person = person

        if best_person is None:
            unassociated.append(ppe)
            continue

        compliance = compliance_by_person[best_person.track_id]
        ppe_type = POSITIVE_PPE.get(ppe.class_name) or NEGATIVE_PPE.get(ppe.class_name)
        if ppe_type is None:
            unassociated.append(ppe)
            continue

        if ppe.class_name in POSITIVE_PPE:
            compliance.items[ppe_type] = "yes"
            compliance.positive_detections.append(ppe)
        else:
            # A positive detection takes precedence over a negative one if both
            # are associated with the same person.
            if compliance.items.get(ppe_type) != "yes":
                compliance.items[ppe_type] = "no"
            compliance.negative_detections.append(ppe)

    return list(compliance_by_person.values()), unassociated
