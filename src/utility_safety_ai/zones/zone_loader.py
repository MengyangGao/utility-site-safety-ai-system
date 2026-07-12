"""Load restricted-zone configurations from YAML or JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .zone import Zone

_ROOT_FIELDS = frozenset({"zones"})
_ZONE_FIELDS = frozenset(
    {
        "id",
        "name",
        "risk_level",
        "polygon",
        "coordinate_space",
        "required_ppe",
        "dwell_seconds",
    }
)


def load_zones(path: str | Path | None) -> list[Zone]:
    """Load zones from a YAML or JSON file.

    Args:
        path: Path to the zone configuration file. ``None`` disables zones.

    Returns:
        A list of :class:`Zone` objects.
    """
    if path is None:
        return []

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Zone configuration file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Zone configuration path is not a file: {path}")

    try:
        data = _read_file(path)
    except Exception as exc:
        raise ValueError(f"Could not parse zone configuration {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("Zone configuration root must be a mapping")
    unknown_root_fields = set(data) - _ROOT_FIELDS
    if unknown_root_fields:
        raise ValueError(
            f"Unknown zone configuration fields: {sorted(unknown_root_fields)}"
        )
    raw_zones = data.get("zones", [])
    if not isinstance(raw_zones, list):
        raise ValueError("Zone configuration 'zones' must be a list")

    zones: list[Zone] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw_zones):
        try:
            zone = _parse_zone(item)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid zone at index {index}: {exc}") from exc
        if zone.id in seen_ids:
            raise ValueError(f"Duplicate zone id: {zone.id!r}")
        seen_ids.add(zone.id)
        zones.append(zone)

    return zones


def _read_file(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        import yaml

        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    raise ValueError("Zone configuration must use .yaml, .yml, or .json")


def _parse_zone(item: dict[str, Any]) -> Zone:
    if not isinstance(item, dict):
        raise TypeError("zone entry must be a mapping")
    unknown_fields = set(item) - _ZONE_FIELDS
    if unknown_fields:
        raise ValueError(f"unknown fields: {sorted(unknown_fields)}")
    if "id" not in item:
        raise ValueError("missing required field 'id'")
    if "polygon" not in item:
        raise ValueError("missing required field 'polygon'")
    zone_id = item["id"]
    if not isinstance(zone_id, str):
        raise ValueError("field 'id' must be a string")
    name = item.get("name", zone_id)
    if not isinstance(name, str):
        raise ValueError("field 'name' must be a string")
    risk_level = item.get("risk_level", "high")
    if not isinstance(risk_level, str):
        raise ValueError("field 'risk_level' must be a string")
    coordinate_space = item.get("coordinate_space", "pixels")
    if not isinstance(coordinate_space, str):
        raise ValueError("field 'coordinate_space' must be a string")
    return Zone(
        id=zone_id,
        name=name,
        risk_level=risk_level,
        polygon=item["polygon"],
        coordinate_space=coordinate_space,
        required_ppe=tuple(item.get("required_ppe", ())),
        dwell_seconds=item.get("dwell_seconds", 0.0),
    )
