"""Restricted-zone dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .geometry import point_in_polygon, scale_polygon

COORDINATE_SPACES = frozenset({"pixels", "normalized"})
PPE_TYPES = frozenset({"helmet", "vest", "gloves", "boots", "goggles"})
RISK_LEVELS = frozenset({"low", "medium", "high", "critical"})


@dataclass(frozen=True)
class Zone:
    """A polygonal restricted zone with an explicit coordinate space.

    ``normalized`` polygons use coordinates in the inclusive ``[0, 1]`` range
    and require a ``(width, height)`` frame size when evaluated.
    """

    id: str
    name: str
    risk_level: str
    polygon: list[tuple[float, float]]
    coordinate_space: str = "pixels"
    required_ppe: tuple[str, ...] = ()
    dwell_seconds: float = 0.0

    def __post_init__(self) -> None:
        zone_id = self.id.strip() if isinstance(self.id, str) else ""
        name = self.name.strip() if isinstance(self.name, str) else ""
        if not zone_id:
            raise ValueError("Zone id must be a non-empty string")
        if not name:
            raise ValueError(f"Zone {zone_id!r} name must be a non-empty string")
        if self.risk_level not in RISK_LEVELS:
            raise ValueError(
                f"Zone {zone_id!r} risk_level must be one of {sorted(RISK_LEVELS)}"
            )
        if self.coordinate_space not in COORDINATE_SPACES:
            raise ValueError(
                "Zone coordinate_space must be either 'pixels' or 'normalized'"
            )
        if not isinstance(self.polygon, (list, tuple)) or len(self.polygon) < 3:
            raise ValueError(f"Zone {zone_id!r} polygon must contain at least 3 points")

        polygon: list[tuple[float, float]] = []
        for index, point in enumerate(self.polygon):
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError(
                    f"Zone {zone_id!r} polygon point {index} must contain x and y"
                )
            x, y = point
            if isinstance(x, bool) or isinstance(y, bool):
                raise ValueError(
                    f"Zone {zone_id!r} polygon point {index} must be numeric"
                )
            try:
                normalized_point = (float(x), float(y))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Zone {zone_id!r} polygon point {index} must be numeric"
                ) from exc
            if not all(isfinite(value) for value in normalized_point):
                raise ValueError(
                    f"Zone {zone_id!r} polygon point {index} must be finite"
                )
            if self.coordinate_space == "normalized" and not all(
                0.0 <= value <= 1.0 for value in normalized_point
            ):
                raise ValueError(
                    f"Zone {zone_id!r} normalized coordinates must be in [0, 1]"
                )
            polygon.append(normalized_point)

        area_twice = abs(
            sum(
                x1 * y2 - x2 * y1
                for (x1, y1), (x2, y2) in zip(
                    polygon, polygon[1:] + polygon[:1], strict=True
                )
            )
        )
        if area_twice <= 1e-12:
            raise ValueError(f"Zone {zone_id!r} polygon must have non-zero area")

        if isinstance(self.required_ppe, str) or not isinstance(
            self.required_ppe, (list, tuple)
        ):
            raise ValueError(f"Zone {zone_id!r} required_ppe must be a list")
        if any(not isinstance(item, str) for item in self.required_ppe):
            raise ValueError(
                f"Zone {zone_id!r} required_ppe values must be strings"
            )
        required_ppe = tuple(dict.fromkeys(self.required_ppe))
        invalid_ppe = set(required_ppe) - PPE_TYPES
        if invalid_ppe:
            raise ValueError(
                f"Zone {zone_id!r} has unsupported required PPE: {sorted(invalid_ppe)}"
            )
        if isinstance(self.dwell_seconds, bool) or not isinstance(
            self.dwell_seconds, (int, float)
        ):
            raise ValueError(f"Zone {zone_id!r} dwell_seconds must be numeric")
        dwell_seconds = float(self.dwell_seconds)
        if not isfinite(dwell_seconds) or dwell_seconds < 0:
            raise ValueError(
                f"Zone {zone_id!r} dwell_seconds must be a finite non-negative value"
            )

        object.__setattr__(self, "id", zone_id)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "polygon", polygon)
        object.__setattr__(self, "required_ppe", required_ppe)
        object.__setattr__(self, "dwell_seconds", dwell_seconds)

    def resolved_polygon(
        self,
        frame_size: tuple[float, float] | None = None,
    ) -> list[tuple[float, float]]:
        """Return pixel coordinates, scaling normalized zones when necessary."""
        if self.coordinate_space == "pixels":
            return list(self.polygon)
        if frame_size is None:
            raise ValueError(
                f"Normalized zone {self.id!r} requires frame_size=(width, height)"
            )
        return scale_polygon(self.polygon, frame_size)

    def contains(
        self,
        point: tuple[float, float],
        frame_size: tuple[float, float] | None = None,
    ) -> bool:
        """Return whether a pixel-space point lies in this zone."""
        return point_in_polygon(point, self.resolved_polygon(frame_size))
