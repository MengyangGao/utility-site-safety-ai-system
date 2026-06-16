"""Load restricted-zone configurations from YAML or JSON files."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .zone import Zone

logger = logging.getLogger(__name__)


def load_zones(path: str | Path | None) -> list[Zone]:
    """Load zones from a YAML or JSON file.

    Args:
        path: Path to the zone configuration file. If None or missing,
            an empty list is returned.

    Returns:
        A list of :class:`Zone` objects.
    """
    if path is None:
        logger.warning("No zone file provided; using empty zone list.")
        return []

    path = Path(path)
    if not path.exists():
        logger.warning("Zone file not found: %s; using empty zone list.", path)
        return []

    try:
        data = _read_file(path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to read zone file %s: %s; using empty zone list.", path, exc)
        return []

    zones: list[Zone] = []
    for item in data.get("zones", []):
        try:
            zones.append(_parse_zone(item))
        except Exception as exc:  # pragma: no cover
            logger.warning("Skipping invalid zone entry: %s; error: %s", item, exc)

    return zones


def _read_file(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        import yaml

        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_zone(item: dict[str, Any]) -> Zone:
    polygon = [tuple(float(v) for v in point) for point in item["polygon"]]
    return Zone(
        id=str(item["id"]),
        name=str(item.get("name", item["id"])),
        risk_level=str(item.get("risk_level", "high")),
        polygon=polygon,
    )
