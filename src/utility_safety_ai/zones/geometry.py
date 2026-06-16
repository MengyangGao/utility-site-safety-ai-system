"""Lightweight geometry helpers for restricted-zone checks."""

from __future__ import annotations


def point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test.

    Args:
        point: (x, y) coordinate.
        polygon: Ordered list of (x, y) vertices. The polygon is closed
            automatically; repeating the first vertex is optional.

    Returns:
        True if the point is inside (or on the edge of) the polygon.
    """
    if not polygon:
        return False

    x, y = point
    inside = False
    n = len(polygon)

    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        # Check if point lies exactly on the edge.
        if _point_on_segment(point, (x1, y1), (x2, y2)):
            return True

        # Standard ray-casting crossing test.
        if ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
        ):
            inside = not inside

    return inside


def _point_on_segment(
    point: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> bool:
    """Return True if point lies on the closed segment ab."""
    x, y = point
    x1, y1 = a
    x2, y2 = b

    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    if abs(cross) > 1e-9:
        return False

    min_x, max_x = sorted((x1, x2))
    min_y, max_y = sorted((y1, y2))
    return min_x - 1e-9 <= x <= max_x + 1e-9 and min_y - 1e-9 <= y <= max_y + 1e-9


def bottom_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    """Return the bottom-center point of a bounding box."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y2)
