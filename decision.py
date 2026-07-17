"""Small, calibration-friendly decision rule for TwinLens."""

LABEL_A = "A"
LABEL_B = "B"
UNCERTAIN = "uncertain"
OTHER = "other"


def decide(score_a, score_b, quality, detection_confidence, *, accept=0.48, margin=0.06,
           other=0.25, min_quality=25.0, min_detection=0.80):
    """Return A/B only when both absolute score and separation are strong."""
    if detection_confidence < min_detection or quality < min_quality:
        return UNCERTAIN
    best_label, best, second = (LABEL_A, score_a, score_b) if score_a >= score_b else (LABEL_B, score_b, score_a)
    if best < other:
        return OTHER
    if best < accept or best - second < margin:
        return UNCERTAIN
    return best_label
