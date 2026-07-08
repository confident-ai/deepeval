"""Offline regression test for issue #2859.

TaskCompletionMetric's `extract_goal_and_outcome` template references
`{{ tools_called_formatted }}`, but the metric previously passed
`tools_called=test_case.tools_called`. Under strict-undefined rendering this
raised `MetricTemplateInterpolationError` before any LLM call. These tests
render the template directly, so they run fully offline (no API key).
"""

import pytest

from deepeval.metrics.utils import print_tools_called
from deepeval.test_case import ToolCall
from deepeval.templates.resolver import (
    resolve_template,
    MetricTemplateInterpolationError,
)


def _formatted_tools():
    tools = [
        ToolCall(
            name="search_flights",
            input_parameters={"origin": "NYC", "destination": "Berlin"},
        )
    ]
    return print_tools_called(tools)


def test_extract_goal_and_outcome_renders_with_tools_called_formatted():
    """The template must render when given the variable it declares."""
    rendered = resolve_template(
        "metrics",
        "TaskCompletionMetric",
        "extract_goal_and_outcome",
        input="Plan a trip to Berlin",
        actual_output="Here are flights and hotels.",
        tools_called_formatted=_formatted_tools(),
    )
    assert "search_flights" in rendered


def test_extract_goal_and_outcome_missing_variable_raises():
    """Passing the old (wrong) kwarg name reproduces the reported error."""
    with pytest.raises(MetricTemplateInterpolationError):
        resolve_template(
            "metrics",
            "TaskCompletionMetric",
            "extract_goal_and_outcome",
            input="Plan a trip to Berlin",
            actual_output="Here are flights and hotels.",
            tools_called=_formatted_tools(),
        )