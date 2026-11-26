"""Exercise the existing plain-test-case path through public metric methods.

The extraction-only regressions cover template interpolation. These tests also
exercise construction, validation, model calls and the final score, without
requiring a provider or populating a private trace field.
"""

import asyncio

import pytest

from deepeval.metrics import TaskCompletionMetric
from deepeval.metrics.task_completion.schema import (
    TaskAndOutcome,
    TaskCompletionVerdict,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, ToolCall


class _SchemaJudge(DeepEvalBaseLLM):
    """Return fixed responses at the model boundary, keeping metric code real."""

    def __init__(self):
        self.calls = []
        super().__init__(model="task-completion-offline-judge")

    def load_model(self):
        return None

    def get_model_name(self):
        return "task-completion-offline-judge"

    def _respond(self, prompt, schema, mode):
        self.calls.append((mode, prompt, schema))
        if schema is TaskAndOutcome:
            return TaskAndOutcome(
                task="PUBLIC_API_EXTRACTED_TASK_6d42",
                outcome="PUBLIC_API_EXTRACTED_OUTCOME_6d42",
            )
        if schema is TaskCompletionVerdict:
            return TaskCompletionVerdict(
                verdict=0.75,
                reason="The observed outcome mostly satisfies the task.",
            )
        raise AssertionError(f"Unexpected response schema: {schema}")

    def generate(self, prompt, schema=None):
        return self._respond(prompt, schema, "sync")

    async def a_generate(self, prompt, schema=None):
        return self._respond(prompt, schema, "async")


@pytest.mark.parametrize("entrypoint", ["sync", "sync_async", "async"])
@pytest.mark.parametrize("tools", ["populated", "empty", "none"])
def test_public_measure_accepts_plain_test_case(entrypoint, tools):
    judge = _SchemaJudge()
    metric = TaskCompletionMetric(
        model=judge,
        eval_mode="llm",
        threshold=0.7,
        async_mode=entrypoint == "sync_async",
    )
    tool_calls = [
        ToolCall(
            name="public_api_dispatch_6d42",
            input_parameters={"recipient": "public-api-6d42@example.test"},
            output={"delivery_id": "PUBLIC_API_DELIVERY_6d42"},
        )
    ]
    tools_called = {"populated": tool_calls, "empty": [], "none": None}[tools]
    test_case = LLMTestCase(
        input="PUBLIC_API_REQUEST_6d42: deliver the project update.",
        actual_output="PUBLIC_API_OUTPUT_6d42: the update was delivered.",
        tools_called=tools_called,
    )

    assert test_case._trace_dict is None
    if entrypoint == "async":
        score = asyncio.run(metric.a_measure(test_case, _show_indicator=False))
    else:
        score = metric.measure(test_case, _show_indicator=False)

    assert test_case._trace_dict is None
    assert score == 0.75
    assert metric.score == 0.75
    assert metric.success is True
    assert metric.reason == "The observed outcome mostly satisfies the task."
    assert [call[2] for call in judge.calls] == [
        TaskAndOutcome,
        TaskCompletionVerdict,
    ]
    expected_mode = "sync" if entrypoint == "sync" else "async"
    assert [call[0] for call in judge.calls] == [expected_mode, expected_mode]

    extraction_prompt = judge.calls[0][1]
    assert test_case.input in extraction_prompt
    assert test_case.actual_output in extraction_prompt
    if tools == "populated":
        assert tool_calls[0].name in extraction_prompt
        assert "public-api-6d42@example.test" in extraction_prompt
        assert "PUBLIC_API_DELIVERY_6d42" in extraction_prompt
    else:
        assert "public_api_dispatch_6d42" not in extraction_prompt
    assert "{{" not in extraction_prompt

    verdict_prompt = judge.calls[1][1]
    assert "PUBLIC_API_EXTRACTED_TASK_6d42" in verdict_prompt
    assert "PUBLIC_API_EXTRACTED_OUTCOME_6d42" in verdict_prompt
