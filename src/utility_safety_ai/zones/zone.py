"""Restricted-zone dataclass."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Zone:
    """A polygonal restricted zone with a configurable risk level."""

    id: str
    name: str
    risk_level: str
    polygon: list[tuple[float, float]]
