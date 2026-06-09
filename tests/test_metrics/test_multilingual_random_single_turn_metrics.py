from __future__ import annotations

import os
import random

import pytest

from tests.test_metrics.multilingual_metric_survey import (
    llm_test_case,
    make_single_turn_metric,
    single_turn_metrics,
)

pytestmark = pytest.mark.skipif(
    os.getenv("OPENAI_API_KEY") is None
    or not os.getenv("OPENAI_API_KEY").strip(),
    reason="OPENAI_API_KEY is not set",
)

_CAR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images/car.png")


def test_random_single_turn_metric_measure_after_translation_smoke():
    metrics = single_turn_metrics()
    assert metrics

    metric_name, metric_cls = random.SystemRandom().choice(metrics)
    metric = make_single_turn_metric(metric_cls)
    assert metric is not None, metric_name

    if getattr(metric, "requires_trace", False):
        pytest.skip(f"{metric_name} requires trace context")

    metric.measure(
        llm_test_case(metric_name, car_path=_CAR),
        _show_indicator=False,
        _log_metric_to_confident=False,
    )
    assert metric.score is not None, metric_name
    assert metric.reason is not None, metric_name
