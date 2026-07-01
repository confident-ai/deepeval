import pytest

import deepeval.metrics.task_completion.task_completion as task_completion_module
from deepeval.metrics import TaskCompletionMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, ToolCall


class FakeLLM(DeepEvalBaseLLM):
    def load_model(self, *args, **kwargs):
        return self

    def generate(self, *args, **kwargs):
        return "{}"

    async def a_generate(self, *args, **kwargs):
        return "{}"

    def get_model_name(self, *args, **kwargs):
        return "fake-llm"


def make_test_case():
    return LLMTestCase(
        input="Plan a weather-aware picnic",
        actual_output="Checked the weather and suggested a park picnic.",
        tools_called=[
            ToolCall(
                name="weather_lookup",
                input_parameters={"city": "San Francisco"},
                output={"forecast": "sunny"},
            )
        ],
    )


def assert_prompt_formats_tools_called(prompt):
    assert "weather_lookup" in prompt
    assert "San Francisco" in prompt
    assert "{{ tools_called_formatted }}" not in prompt


def test_task_completion_goal_prompt_formats_tools_called_sync(monkeypatch):
    captured = {}

    def fake_generate_with_schema_and_extract(**kwargs):
        captured["prompt"] = kwargs["prompt"]
        return "task", "outcome"

    monkeypatch.setattr(
        task_completion_module,
        "generate_with_schema_and_extract",
        fake_generate_with_schema_and_extract,
    )

    metric = TaskCompletionMetric(model=FakeLLM(), async_mode=False)

    assert metric._extract_task_and_outcome(make_test_case()) == (
        "task",
        "outcome",
    )
    assert_prompt_formats_tools_called(captured["prompt"])


@pytest.mark.asyncio
async def test_task_completion_goal_prompt_formats_tools_called_async(
    monkeypatch,
):
    captured = {}

    async def fake_a_generate_with_schema_and_extract(**kwargs):
        captured["prompt"] = kwargs["prompt"]
        return "task", "outcome"

    monkeypatch.setattr(
        task_completion_module,
        "a_generate_with_schema_and_extract",
        fake_a_generate_with_schema_and_extract,
    )

    metric = TaskCompletionMetric(model=FakeLLM())

    assert await metric._a_extract_task_and_outcome(make_test_case()) == (
        "task",
        "outcome",
    )
    assert_prompt_formats_tools_called(captured["prompt"])
