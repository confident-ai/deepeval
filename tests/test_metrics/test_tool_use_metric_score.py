"""
Regression tests for ToolUseMetric._calculate_score().

Argument correctness is only scored for interactions that actually called a
tool. When none did, the empty list used to be averaged as 0.0 and folded into
a min(), so a conversation that correctly needed no tools scored 0 -- the same
score as a model that picked every tool wrong.

Offline: `__init__` is bypassed so no model, network or API key is required.
"""

import pytest

from deepeval.metrics.tool_use.tool_use import ToolUseMetric
from deepeval.metrics.tool_use.schema import (
    ToolSelectionScore,
    ArgumentCorrectnessScore,
)


def _metric(strict_mode=False, threshold=0.5):
    # Bypass __init__, which builds an LLM client; _calculate_score only needs
    # these two attributes.
    metric = ToolUseMetric.__new__(ToolUseMetric)
    metric.strict_mode = strict_mode
    metric.threshold = threshold
    return metric


def _selection(*scores):
    return [ToolSelectionScore(score=s, reason="reason") for s in scores]


def _arguments(*scores):
    return [ArgumentCorrectnessScore(score=s, reason="reason") for s in scores]


def test_no_tools_used_falls_back_to_the_tool_selection_score():
    # The model correctly decided no tool was needed. There are no tool
    # arguments to judge, so the score is the tool selection score.
    assert _metric()._calculate_score(_selection(1.0), []) == 1.0


@pytest.mark.parametrize("selection_score", [0.0, 0.25, 0.5, 1.0])
def test_no_tools_used_preserves_a_bad_tool_selection_score(selection_score):
    # Falling back must not mask a wrong selection decision, e.g. the model
    # should have called a tool and did not.
    assert (
        _metric()._calculate_score(_selection(selection_score), [])
        == selection_score
    )


def test_tools_used_still_takes_the_minimum():
    assert _metric()._calculate_score(_selection(1.0), _arguments(0.4)) == 0.4
    assert _metric()._calculate_score(_selection(0.2), _arguments(0.9)) == 0.2


def test_argument_correctness_is_averaged_over_the_turns_that_used_tools():
    # Two of three interactions used a tool; the average is over those two.
    score = _metric()._calculate_score(
        _selection(1.0, 1.0, 1.0), _arguments(1.0, 0.5)
    )
    assert score == 0.75


def test_strict_mode_still_zeroes_a_below_threshold_score():
    assert (
        _metric(strict_mode=True, threshold=1)._calculate_score(
            _selection(0.9), []
        )
        == 0
    )


def test_strict_mode_passes_a_perfect_no_tool_conversation():
    assert (
        _metric(strict_mode=True, threshold=1)._calculate_score(
            _selection(1.0), []
        )
        == 1.0
    )
