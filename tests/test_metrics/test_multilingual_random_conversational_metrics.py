from __future__ import annotations

import os
import random

import pytest

from tests.test_metrics.multilingual_metric_survey_conversational import (
    conversational_test_case,
    conversational_metrics,
    make_conversational_metric,
)

pytestmark = pytest.mark.skipif(
    os.getenv("OPENAI_API_KEY") is None
    or not os.getenv("OPENAI_API_KEY").strip(),
    reason="OPENAI_API_KEY is not set",
)


def test_random_conversational_metric_measure_after_translation_smoke():
    metrics = conversational_metrics()
    assert metrics

    metric_name, metric_cls = random.SystemRandom().choice(metrics)
    metric = make_conversational_metric(metric_cls)
    assert metric is not None, metric_name

    metric.measure(
        conversational_test_case(),
        _show_indicator=False,
        _log_metric_to_confident=False,
    )
    assert metric.score is not None, metric_name
    assert metric.reason is not None, metric_name
