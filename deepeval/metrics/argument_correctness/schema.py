"""Deprecated: moved to ``deepeval.metrics.argument_relevancy.schema``."""

from deepeval.metrics.argument_relevancy.schema import (
    ArgumentRelevancyScoreReason as ArgumentCorrectnessScoreReason,
    ArgumentRelevancyVerdict as ArgumentCorrectnessVerdict,
    Verdicts,
)

__all__ = [
    "ArgumentCorrectnessScoreReason",
    "ArgumentCorrectnessVerdict",
    "Verdicts",
]
