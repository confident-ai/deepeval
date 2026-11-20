"""Sync/async parity for ``generate_with_schema_and_extract`` when a custom
(non-native) model returns ``(result, cost)`` from ``generate``/``a_generate``.

The async helper already unpacks such tuples and accrues the cost; the sync
helper used to pass the tuple straight into ``trimAndLoadJson``, which crashed
with ``AttributeError: 'tuple' object has no attribute 'find'``. These tests
exercise the real call site (``ToolUseMetric``) with a stub model, so no API
key is needed.
"""

import pytest

from deepeval.metrics import ToolUseMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import ConversationalTestCase, ToolCall, Turn

COST_PER_CALL = 0.01
SCORE_JSON = '{"score": 1, "reason": "stub"}'
# tool selection, argument correctness, and one final reason per score list
EXPECTED_CALLS = 4


class TupleCostModel(DeepEvalBaseLLM):
    """Custom (non-native) model whose sync and async generation both return
    ``(json_text, cost)`` — the custom-model return shape the async path of
    ``generate_with_schema_and_extract`` already supports."""

    def load_model(self):
        return None

    def get_model_name(self):
        return "tuple-cost-stub"

    def generate(self, prompt, schema=None, **kwargs):
        return (SCORE_JSON, COST_PER_CALL)

    async def a_generate(self, prompt, schema=None, **kwargs):
        return (SCORE_JSON, COST_PER_CALL)


def _make_metric(async_mode: bool) -> ToolUseMetric:
    return ToolUseMetric(
        available_tools=[ToolCall(name="book_flight")],
        model=TupleCostModel(),
        async_mode=async_mode,
    )


def _convo() -> ConversationalTestCase:
    return ConversationalTestCase(
        turns=[
            Turn(role="user", content="Book me a flight to Paris"),
            Turn(
                role="assistant",
                content="Booking your flight now.",
                tools_called=[ToolCall(name="book_flight")],
            ),
        ]
    )


def _record_accrued_costs(metric: ToolUseMetric) -> list:
    """Capture every cost handed to ``metric._accrue_cost`` during a measure."""
    accrued = []
    original = metric._accrue_cost

    def recording(cost):
        accrued.append(cost)
        original(cost)

    metric._accrue_cost = recording
    return accrued


def test_async_measure_with_tuple_cost_model():
    metric = _make_metric(async_mode=True)
    accrued = _record_accrued_costs(metric)

    # measure() with async_mode=True runs a_measure on a loop and reports
    # the score through the attribute.
    metric.measure(_convo())

    assert metric.score == 1.0
    assert len(accrued) == EXPECTED_CALLS
    assert sum(accrued) == pytest.approx(EXPECTED_CALLS * COST_PER_CALL)


def test_sync_measure_with_tuple_cost_model_matches_async():
    async_metric = _make_metric(async_mode=True)
    async_accrued = _record_accrued_costs(async_metric)
    async_metric.measure(_convo())

    sync_metric = _make_metric(async_mode=False)
    sync_accrued = _record_accrued_costs(sync_metric)
    # Crashed with AttributeError before the sync helper learned to unpack
    # the (result, cost) tuple.
    sync_metric.measure(_convo())

    assert sync_metric.score == async_metric.score == 1.0
    assert sync_metric.reason == async_metric.reason
    # The sync path accrues the same per-call costs as the async path.
    assert len(sync_accrued) == len(async_accrued) == EXPECTED_CALLS
    assert sum(sync_accrued) == pytest.approx(EXPECTED_CALLS * COST_PER_CALL)
