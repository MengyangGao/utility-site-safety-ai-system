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
