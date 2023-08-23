"""Tests for toxic
"""


import pytest
from deepeval.metrics.bias_classifier import UnBiasedMetric, assert_unbiased

generated_text = "Winners of the FIFA world cup were the French national football team"


@pytest.mark.asyncio
async def test_non_toxic():
    assert_unbiased(
        text="Who won the FIFA World Cup in 2018?",
    )


@pytest.mark.asyncio
async def test_non_toxic_2():
    with pytest.raises(AssertionError):
        assert_unbiased(
            text="Left wing is evil.",
        )


@pytest.mark.asyncio
async def test_non_toxic_metric():
    metric = UnBiasedMetric()
    score = metric.measure(text=generated_text)
    assert metric.is_successful(), "Overall score metric not working"
