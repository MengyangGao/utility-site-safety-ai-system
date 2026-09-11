"""Model promotion gates must fail transparently on missing or weak classes."""

from utility_safety_ai.training.quality_gate import assess_model_metrics


def test_quality_gate_passes_complete_strong_metrics():
    result = assess_model_metrics(
        {
            "precision": 0.8,
            "recall": 0.75,
            "map50": 0.72,
            "per_class": {"no_helmet": {"recall": 0.7}},
        },
        required_classes=("no_helmet",),
    )

    assert result["passed"] is True
    assert result["failures"] == []


def test_quality_gate_exposes_missing_and_zero_recall_classes():
    result = assess_model_metrics(
        {
            "precision": 0.7,
            "recall": 0.55,
            "map50": 0.61,
            "per_class": {"no_boots": {"recall": 0.0}},
        },
        required_classes=("no_boots", "no_vest"),
    )

    assert result["passed"] is False
    assert any("no_boots" in failure for failure in result["failures"])
    assert any("no_vest" in failure for failure in result["failures"])


def test_quality_gate_rejects_nan_instead_of_promoting_a_model():
    result = assess_model_metrics(
        {"precision": float("nan"), "recall": float("nan"), "map50": float("nan")}
    )
    assert not result["passed"]
    assert len(result["failures"]) == 3


def test_quality_gate_rejects_missing_unmeasured_and_out_of_range_values():
    import pytest

    for value in (None, float("inf"), -0.1, 1.1, True, "0.9"):
        result = assess_model_metrics({"precision": value, "recall": 0.8, "map50": 0.8})
        assert not result["passed"]
    with pytest.raises(ValueError):
        assess_model_metrics({}, min_recall=float("nan"))
    result = assess_model_metrics(
        {
            "precision": 0.8,
            "recall": 0.8,
            "map50": 0.8,
            "per_class": {"no_helmet": {"recall": None}},
        },
        required_classes=("no_helmet",),
    )
    assert not result["passed"]


def test_validation_maps_sparse_class_ids_and_marks_absent_classes():
    from types import SimpleNamespace

    from utility_safety_ai.tools.validate_ppe_model import class_metrics

    metrics = SimpleNamespace(
        names={0: "helmet", 1: "gloves", 2: "person"},
        box=SimpleNamespace(
            ap_class_index=[0, 2], p=[0.7, 0.9], r=[0.6, 0.8], ap50=[0.5, 0.7], ap=[0.4, 0.6]
        ),
    )
    report = class_metrics(metrics)
    assert report["person"]["precision"] == 0.9
    assert report["gloves"]["recall"] is None
    assert report["gloves"]["evaluated"] is False
