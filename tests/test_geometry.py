"""Tests for zone geometry helpers."""

from __future__ import annotations

from utility_safety_ai.zones.geometry import (
    bottom_center,
    point_in_polygon,
    scale_polygon,
)

SQUARE = [(0, 0), (10, 0), (10, 10), (0, 10)]
TRIANGLE = [(0, 0), (10, 0), (5, 10)]


def test_point_inside_square():
    assert point_in_polygon((5, 5), SQUARE) is True


def test_point_outside_square():
    assert point_in_polygon((15, 5), SQUARE) is False


def test_point_on_edge_counts_as_inside():
    assert point_in_polygon((0, 5), SQUARE) is True


def test_point_inside_triangle():
    assert point_in_polygon((5, 2), TRIANGLE) is True


def test_empty_polygon_returns_false():
    assert point_in_polygon((0, 0), []) is False


def test_bottom_center():
    assert bottom_center((0, 0, 10, 20)) == (5, 20)


def test_scale_normalized_polygon():
    assert scale_polygon([(0.25, 0.5), (1.0, 1.0)], (200, 100)) == [
        (50, 50),
        (200, 100),
    ]
