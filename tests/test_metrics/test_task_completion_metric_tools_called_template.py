"""Regression tests for TaskCompletionMetric tool-call prompt rendering.

These tests exercise the legacy no-trace fallback path directly and do not
require an API key.
"""

import asyncio
from unittest.mock import patch

from deepeval.metrics import TaskCompletionMetric
from deepeval.test_case import LLMTestCase, ToolCall
from tests.test_core.stubs import DummyModel


def make_metric(*, async_mode: bool = False) -> TaskCompletionMetric:
    with patch(
        "deepeval.metrics.task_completion.task_completion.initialize_model"
    ) as mock_init:
        mock_init.return_value = (DummyModel(), True)
        return TaskCompletionMetric(async_mode=async_mode)


def make_test_case() -> LLMTestCase:
    return LLMTestCase(
        input="Find a flight from NYC to Berlin on March 15.",
        actual_output="I found matching flights from NYC to Berlin.",
        tools_called=[
            ToolCall(
                name="search_flights",
                input_parameters={
                    "origin": "NYC",
                    "destination": "Berlin",
                    "date": "2025-03-15",
                },
            )
        ],
    )


def test_task_completion_no_trace_formats_tools_called_sync():
    metric = make_metric(async_mode=False)
    test_case = make_test_case()
    captured = {}

    def fake_generate_with_schema_and_extract(**kwargs):
        captured["prompt"] = kwargs["prompt"]
        return "Book travel", "The agent searched for flights."

    assert test_case._trace_dict is None

    with patch(
        "deepeval.metrics.task_completion.task_completion.generate_with_schema_and_extract",
        side_effect=fake_generate_with_schema_and_extract,
    ):
        task, outcome = metric._extract_task_and_outcome(test_case)

    assert task == "Book travel"
    assert outcome == "The agent searched for flights."
    assert "search_flights" in captured["prompt"]
    assert '"origin": "NYC"' in captured["prompt"]


def test_task_completion_no_trace_formats_tools_called_async():
    asyncio.run(_run_async_extract_task_and_outcome_test())


async def _run_async_extract_task_and_outcome_test():
    metric = make_metric(async_mode=True)
    test_case = make_test_case()
    captured = {}

    async def fake_a_generate_with_schema_and_extract(**kwargs):
        captured["prompt"] = kwargs["prompt"]
        return "Book travel", "The agent searched for flights."

    assert test_case._trace_dict is None

    with patch(
        "deepeval.metrics.task_completion.task_completion.a_generate_with_schema_and_extract",
        side_effect=fake_a_generate_with_schema_and_extract,
    ):
        task, outcome = await metric._a_extract_task_and_outcome(test_case)

    assert task == "Book travel"
    assert outcome == "The agent searched for flights."
    assert "search_flights" in captured["prompt"]
    assert '"origin": "NYC"' in captured["prompt"]
