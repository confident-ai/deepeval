"""Regression test for the TaskCompletionMetric `tools_called` -> jinja
`tools_called_formatted` mismatch (issue #2807).

The non-trace branch of `TaskCompletionMetric._extract_task_and_outcome` /
`_a_extract_task_and_outcome` renders the `extract_goal_and_outcome` template,
which references `{{ tools_called_formatted }}`. Previously the metric passed
`tools_called=test_case.tools_called` (the wrong kwarg, and an un-stringified
`List[ToolCall]`), so under `jinja2.StrictUndefined` the render raised
`MetricTemplateInterpolationError` and the formatted tool list never reached
the prompt.

The tool names used below (`quantum_router`, `cryo_vault`) are deliberately
absent from the template's own static example block, so a positive
`"quantum_router" in prompt` assertion can only pass if the value the metric
formats actually reaches the render -- it cannot be satisfied by template
boilerplate. This keeps the assertions non-vacuous even against a future
silent "renders to empty string" regression.
"""

import asyncio

import pytest

import deepeval.metrics.task_completion.task_completion as task_completion_module
from deepeval.metrics import TaskCompletionMetric
from deepeval.metrics.utils import print_tools_called
from deepeval.templates.resolver import (
    MetricTemplateInterpolationError,
    clear_metric_template_cache,
)
from deepeval.test_case import LLMTestCase, ToolCall

# Names chosen so they do NOT appear anywhere in the template's static example
# text -- their presence in the rendered prompt proves the user's tools_called
# reached the template.
_TOOL_A = "quantum_router"
_TOOL_B = "cryo_vault"


def _make_test_case() -> LLMTestCase:
    return LLMTestCase(
        input="Can you help me plan a trip to New York this weekend?",
        actual_output="Sure! Here are some flights and hotels.",
        tools_called=[
            ToolCall(name=_TOOL_A),
            ToolCall(name=_TOOL_B),
        ],
    )


def _metric_with_captured_prompt(monkeypatch):
    """Build a TaskCompletionMetric whose extract path is exercised without a
    real model call, capturing the prompt handed to the (a_)generate helper."""
    metric = TaskCompletionMetric.__new__(TaskCompletionMetric)
    metric.task = None
    metric._is_task_provided = False

    captured = {}

    def _fake_generate(*, metric, prompt, schema_cls, extract_schema, **kwargs):
        captured["prompt"] = prompt
        return ("the task", "the outcome")

    async def _fake_a_generate(
        *, metric, prompt, schema_cls, extract_schema, **kwargs
    ):
        captured["prompt"] = prompt
        return ("the task", "the outcome")

    monkeypatch.setattr(
        task_completion_module,
        "generate_with_schema_and_extract",
        _fake_generate,
    )
    monkeypatch.setattr(
        task_completion_module,
        "a_generate_with_schema_and_extract",
        _fake_a_generate,
    )
    return metric, captured


def test_sync_extract_includes_tools_called(monkeypatch):
    """Regression proof for the call-site fix (sync path). Fails on the buggy
    code with MetricTemplateInterpolationError before any prompt is produced."""
    clear_metric_template_cache()
    metric, captured = _metric_with_captured_prompt(monkeypatch)

    task, outcome = metric._extract_task_and_outcome(_make_test_case())

    assert (task, outcome) == ("the task", "the outcome")
    prompt = captured["prompt"]
    # Non-vacuous: these names exist only in the user's tools_called, never in
    # the template boilerplate, so they can only appear via a real render.
    assert _TOOL_A in prompt
    assert _TOOL_B in prompt
    # The unrendered jinja variable must not leak into the prompt.
    assert "tools_called_formatted" not in prompt
    assert "{{" not in prompt


def test_async_extract_includes_tools_called(monkeypatch):
    """Regression proof for the call-site fix (async path)."""
    clear_metric_template_cache()
    metric, captured = _metric_with_captured_prompt(monkeypatch)

    task, outcome = asyncio.run(
        metric._a_extract_task_and_outcome(_make_test_case())
    )

    assert (task, outcome) == ("the task", "the outcome")
    prompt = captured["prompt"]
    assert _TOOL_A in prompt
    assert _TOOL_B in prompt
    assert "tools_called_formatted" not in prompt
    assert "{{" not in prompt


def test_buggy_kwarg_raises_under_strict_undefined():
    """Document the root cause: the template requires `tools_called_formatted`
    under StrictUndefined, so building the prompt WITHOUT it (the pre-fix call
    shape, which passed `tools_called=`) raises. Guards against anyone
    re-rendering this template without supplying the formatted kwarg."""
    clear_metric_template_cache()
    metric = TaskCompletionMetric.__new__(TaskCompletionMetric)

    with pytest.raises(MetricTemplateInterpolationError):
        metric._get_prompt(
            "extract_goal_and_outcome",
            input="plan a trip",
            actual_output="done",
            # tools_called_formatted intentionally omitted -> the
            # StrictUndefined render must fail, exactly as in issue #2807.
        )


def test_extract_goal_and_outcome_template_renders_tool_names():
    """Template-contract check (NOT the call-site fix proof -- tests 1 and 2
    are that). Given the correct `tools_called_formatted` kwarg directly, the
    template renders without a StrictUndefined error and embeds the tool
    names."""
    clear_metric_template_cache()
    metric = TaskCompletionMetric.__new__(TaskCompletionMetric)
    tools_called = [ToolCall(name=_TOOL_A), ToolCall(name=_TOOL_B)]

    prompt = metric._get_prompt(
        "extract_goal_and_outcome",
        input="plan a trip",
        actual_output="done",
        tools_called_formatted=print_tools_called(tools_called),
    )

    assert _TOOL_A in prompt
    assert _TOOL_B in prompt
