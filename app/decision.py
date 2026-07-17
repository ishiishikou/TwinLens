from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class DecisionLabel(StrEnum):
    TWIN_A = "TWIN_A"
    TWIN_B = "TWIN_B"
    INDETERMINATE = "INDETERMINATE"
    OTHER_OR_NO_FACE = "OTHER_OR_NO_FACE"


@dataclass(frozen=True)
class Decision:
    label: DecisionLabel
    score_a: float
    score_b: float
    best_score: float
    margin: float
    reason: str


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ValueError("zero-length embedding")
    return vector.astype(np.float32) / norm


def aggregate_similarity(query: np.ndarray, references: list[np.ndarray], top_k: int = 5) -> float:
    if not references:
        return -1.0
    q = _normalize(query)
    matrix = np.vstack([_normalize(item) for item in references])
    similarities = matrix @ q
    k = min(top_k, len(similarities))
    nearest_mean = float(np.mean(np.partition(similarities, -k)[-k:]))
    centroid = _normalize(np.mean(matrix, axis=0))
    centroid_score = float(centroid @ q)
    return 0.7 * nearest_mean + 0.3 * centroid_score


def decide(
    *,
    score_a: float,
    score_b: float,
    quality: float,
    detection_confidence: float,
    count_a: int,
    count_b: int,
    min_quality: float,
    min_detection_confidence: float,
    min_references_per_subject: int,
    other_threshold: float,
    accept_threshold: float,
    margin_threshold: float,
) -> Decision:
    best_label = DecisionLabel.TWIN_A if score_a >= score_b else DecisionLabel.TWIN_B
    best = max(score_a, score_b)
    second = min(score_a, score_b)
    margin = best - second

    if count_a < min_references_per_subject or count_b < min_references_per_subject:
        return Decision(DecisionLabel.INDETERMINATE, score_a, score_b, best, margin, "insufficient_reference_images")
    if detection_confidence < min_detection_confidence:
        return Decision(DecisionLabel.INDETERMINATE, score_a, score_b, best, margin, "low_detection_confidence")
    if quality < min_quality:
        return Decision(DecisionLabel.INDETERMINATE, score_a, score_b, best, margin, "low_image_quality")
    if best < other_threshold:
        return Decision(DecisionLabel.OTHER_OR_NO_FACE, score_a, score_b, best, margin, "below_other_threshold")
    if best < accept_threshold:
        return Decision(DecisionLabel.INDETERMINATE, score_a, score_b, best, margin, "below_accept_threshold")
    if margin < margin_threshold:
        return Decision(DecisionLabel.INDETERMINATE, score_a, score_b, best, margin, "insufficient_margin")
    return Decision(best_label, score_a, score_b, best, margin, "accepted")
