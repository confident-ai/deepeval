"""Unit tests for ContextualRecallMetric._calculate_score.

These exercise the pure scoring logic without a model/API key: the metric is
constructed without running ``__init__`` (which would require a judge model),
verdicts are set directly, and the score is computed.
"""

from deepeval.metrics import ContextualRecallMetric
from deepeval.metrics.contextual_recall.schema import ContextualRecallVerdict


def _metric_with_verdicts(verdict_strings):
    metric = ContextualRecallMetric.__new__(ContextualRecallMetric)
    metric.strict_mode = False
    metric.threshold = 0.5
    metric.verdicts = [
        ContextualRecallVerdict(verdict=v, reason="") for v in verdict_strings
    ]
    return metric


def test_calculate_score_strips_whitespace_from_verdicts():
    """Verdicts are counted after stripping surrounding whitespace.

    The judge's "yes" verdicts sometimes arrive padded (e.g. "yes\\n" or " yes").
    They must still count as justified, matching how every other deepeval metric
    normalizes verdicts (".strip().lower()"). Otherwise recall is silently
    undercounted (here 0.25 instead of 0.75).
    """
    metric = _metric_with_verdicts(["yes", "yes\n", " yes", "no"])
    assert metric._calculate_score() == 0.75


def test_calculate_score_plain_verdicts_unchanged():
    metric = _metric_with_verdicts(["yes", "no", "yes", "yes"])
    assert metric._calculate_score() == 0.75
