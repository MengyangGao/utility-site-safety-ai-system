"""Risk level constants."""

from __future__ import annotations

LOW = "low"
MEDIUM = "medium"
HIGH = "high"
CRITICAL = "critical"

RISK_ORDER = {LOW: 0, MEDIUM: 1, HIGH: 2, CRITICAL: 3}


def max_risk(a: str, b: str) -> str:
    """Return the higher-risk level of the two inputs."""
    return a if RISK_ORDER.get(a, -1) >= RISK_ORDER.get(b, -1) else b
