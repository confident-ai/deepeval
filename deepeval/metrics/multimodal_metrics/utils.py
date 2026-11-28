from collections.abc import Sequence


def clamp_image_score(raw_score: float) -> float:
    """Clamp a judge's 0-10 image score to [0.0, 10.0]."""
    return min(max(float(raw_score), 0.0), 10.0)


def clamp_image_scores(raw_scores: Sequence[int | float]) -> list[float]:
    """Clamp a sequence of judge 0-10 image scores to [0.0, 10.0]."""
    return [clamp_image_score(score) for score in raw_scores]
