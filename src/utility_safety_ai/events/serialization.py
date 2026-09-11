"""Shared CSV serialization for portable, spreadsheet-safe reports."""

from __future__ import annotations

import json
from typing import Any


def csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, allow_nan=False)
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return str(value)
