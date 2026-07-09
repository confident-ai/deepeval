"""Offline regression tests for the TaskCompletionMetric template fix (#2807).

The `extract_goal_and_outcome` template references `{{ tools_called_formatted }}`,
but the non-trace path used to pass `tools_called=...`. Because templates render
with strict-undefined, that raised `MetricTemplateInterpolationError` before any
LLM call. These tests render the template directly, so they need no API key.
"""

import pytest

from deepeval.metrics.utils import print_tools_called
from deepeval.templates.resolver import (
    MetricTemplateInterpolationError,
    resolve_template,
)
from deepeval.test_case import ToolCall

_INPUT = "Book a dinner reservation for two tonight"
_OUTPUT = "Reserved a table for two at 8pm."


def _formatted_tools():
    return print_tools_called(
        [ToolCall(name="reserve_table", input_parameters={"party_size": 2})]
    )


def test_template_renders_with_tools_called_formatted():
    """Rendering succeeds when the variable the template declares is supplied."""
    rendered = resolve_template(
        "metrics",
        "TaskCompletionMetric",
        "extract_goal_and_outcome",
        input=_INPUT,
        actual_output=_OUTPUT,
        tools_called_formatted=_formatted_tools(),
    )
    assert "reserve_table" in rendered
    assert _INPUT in rendered


def test_old_kwarg_reproduces_the_render_error():
    """The previous kwarg name leaves the template variable undefined."""
    with pytest.raises(
        MetricTemplateInterpolationError, match="tools_called_formatted"
    ):
        resolve_template(
            "metrics",
            "TaskCompletionMetric",
            "extract_goal_and_outcome",
            input=_INPUT,
            actual_output=_OUTPUT,
            tools_called=_formatted_tools(),
        )


def test_template_renders_with_no_tools():
    """Empty tool calls format to an empty string and still render cleanly."""
    rendered = resolve_template(
        "metrics",
        "TaskCompletionMetric",
        "extract_goal_and_outcome",
        input=_INPUT,
        actual_output=_OUTPUT,
        tools_called_formatted=print_tools_called([]),
    )
    assert _INPUT in rendered
