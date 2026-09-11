"""Deterministic promotion gate for retained model-evaluation metrics."""

from __future__ import annotations

from math import isfinite
from typing import Any


def assess_model_metrics(
    metrics: dict[str, Any],
    *,
    min_precision: float = 0.65,
    min_recall: float = 0.65,
    min_map50: float = 0.60,
    min_class_recall: float = 0.50,
    required_classes: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Return a machine-readable pass/fail result without hiding weak classes."""

    failures: list[str] = []
    for name, minimum in (
        ("precision", min_precision),
        ("recall", min_recall),
        ("map50", min_map50),
    ):
        if isinstance(minimum, bool) or not isfinite(minimum) or not 0 <= minimum <= 1:
            raise ValueError(f"{name} threshold must be finite and in [0, 1]")
        raw = metrics.get(name)
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not isfinite(raw)
            or not 0 <= raw <= 1
        ):
            failures.append(f"{name} must be a finite measured value in [0, 1]")
            continue
        actual = float(raw)
        if actual < minimum:
            failures.append(f"{name}={actual:.4f} is below {minimum:.4f}")

    if (
        isinstance(min_class_recall, bool)
        or not isfinite(min_class_recall)
        or not 0 <= min_class_recall <= 1
    ):
        raise ValueError("class recall threshold must be finite and in [0, 1]")
    per_class = metrics.get("per_class", {})
    if not isinstance(per_class, dict):
        failures.append("per_class metrics are missing")
        per_class = {}
    for class_name in required_classes:
        class_metrics = per_class.get(class_name)
        if not isinstance(class_metrics, dict):
            failures.append(f"required class {class_name!r} is missing")
            continue
        raw = class_metrics.get("recall")
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not isfinite(raw)
            or not 0 <= raw <= 1
        ):
            failures.append(
                f"class {class_name!r} recall must be a finite measured value in [0, 1]"
            )
            continue
        recall = float(raw)
        if recall < min_class_recall:
            failures.append(
                f"class {class_name!r} recall={recall:.4f} is below {min_class_recall:.4f}"
            )

    return {
        "passed": not failures,
        "thresholds": {
            "precision": min_precision,
            "recall": min_recall,
            "map50": min_map50,
            "class_recall": min_class_recall,
        },
        "required_classes": list(required_classes),
        "failures": failures,
    }
