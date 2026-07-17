import numpy as np

from app.decision import DecisionLabel, aggregate_similarity, decide


BASE = dict(
    quality=0.9,
    detection_confidence=0.99,
    count_a=20,
    count_b=20,
    min_quality=0.35,
    min_detection_confidence=0.85,
    min_references_per_subject=3,
    other_threshold=0.28,
    accept_threshold=0.42,
    margin_threshold=0.08,
)


def test_accepts_a_only_with_absolute_score_and_margin() -> None:
    result = decide(score_a=0.63, score_b=0.49, **BASE)
    assert result.label == DecisionLabel.TWIN_A


def test_close_twins_are_indeterminate() -> None:
    result = decide(score_a=0.61, score_b=0.58, **BASE)
    assert result.label == DecisionLabel.INDETERMINATE
    assert result.reason == "insufficient_margin"


def test_low_similarity_is_other() -> None:
    result = decide(score_a=0.12, score_b=0.18, **BASE)
    assert result.label == DecisionLabel.OTHER_OR_NO_FACE


def test_low_quality_is_indeterminate_even_with_high_similarity() -> None:
    result = decide(score_a=0.8, score_b=0.2, **{**BASE, "quality": 0.1})
    assert result.label == DecisionLabel.INDETERMINATE


def test_similarity_uses_references_and_centroid() -> None:
    query = np.array([1.0, 0.0], dtype=np.float32)
    references = [np.array([1.0, 0.0], dtype=np.float32), np.array([0.9, 0.1], dtype=np.float32)]
    assert aggregate_similarity(query, references) > 0.99
