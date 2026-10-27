"""Regression test: sync/async parity in ``*_generate_with_schema_and_extract``.

``a_generate_with_schema_and_extract`` unwraps a ``(text, cost)`` tuple returned
by a non-native (custom) model before handing the payload to ``trimAndLoadJson``
(added in e0684df2, "AttributeError: 'tuple' object has no attribute 'find'").
The synchronous ``generate_with_schema_and_extract`` never got the same
treatment, so a custom ``DeepEvalBaseLLM`` whose ``generate`` returns
``(text, cost)`` works under ``a_measure`` but blows up under ``measure`` with an
opaque ``AttributeError`` raised from inside ``trimAndLoadJson``.

The cost returned in the tuple (``0.25`` below) is deliberately non-zero so the
``_accrue_cost`` assertion can only pass if the tuple was actually unwrapped and
accounted for -- it cannot be satisfied by the default ``0`` accumulator.
"""

import asyncio

import pytest

from pydantic import BaseModel

from deepeval.metrics.utils import (
    a_generate_with_schema_and_extract,
    generate_with_schema_and_extract,
)


class _Verdict(BaseModel):
    reason: str


class _TupleReturningModel:
    """A non-native custom model that returns ``(text, cost)`` -- the shape the
    async helper already tolerates."""

    def generate_with_schema(self, prompt, schema=None):
        return '{"reason": "grounded"}', 0.25

    async def a_generate_with_schema(self, prompt, schema=None):
        return '{"reason": "grounded"}', 0.25


class _Metric:
    using_native_model = False

    def __init__(self):
        self.model = _TupleReturningModel()
        self.accrued_cost = 0.0
        self.error = None

    def _accrue_cost(self, cost):
        self.accrued_cost += cost or 0.0

    def _accrue_tokens(self, input_tokens, output_tokens):
        pass


_KWARGS = dict(
    prompt="irrelevant",
    schema_cls=_Verdict,
    extract_schema=lambda r: r.reason,
    extract_json=lambda data: data["reason"],
)


def test_sync_helper_unwraps_cost_tuple_like_async():
    metric = _Metric()
    result = generate_with_schema_and_extract(metric=metric, **_KWARGS)
    assert result == "grounded"
    assert metric.accrued_cost == pytest.approx(0.25)


def test_async_helper_unwraps_cost_tuple():
    metric = _Metric()
    result = asyncio.run(
        a_generate_with_schema_and_extract(metric=metric, **_KWARGS)
    )
    assert result == "grounded"
    assert metric.accrued_cost == pytest.approx(0.25)


def test_sync_and_async_agree():
    sync_metric, async_metric = _Metric(), _Metric()
    sync_result = generate_with_schema_and_extract(
        metric=sync_metric, **_KWARGS
    )
    async_result = asyncio.run(
        a_generate_with_schema_and_extract(metric=async_metric, **_KWARGS)
    )
    assert sync_result == async_result
    assert sync_metric.accrued_cost == async_metric.accrued_cost
