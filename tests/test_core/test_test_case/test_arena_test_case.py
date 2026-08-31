"""Tests for ArenaTestCase contestant-count validation.

An arena compares at least two contestants, so an empty or single-contestant
arena is a degenerate input: previously it either crashed with an opaque
`IndexError` (empty list indexing ``cases[0]``) or silently slipped through as
a no-op. These tests pin down the new clear failure mode and guard the valid
multi-contestant path against regressions.
"""

import pytest

from deepeval.errors import MissingTestCaseParamsError
from deepeval.metrics.utils import check_arena_test_case_params
from deepeval.test_case import (
    ArenaTestCase,
    Contestant,
    LLMTestCase,
    SingleTurnParams,
)


class DummyArenaMetric:
    __name__ = "DummyArenaMetric"
    error = None


def _contestant(name: str, input_: str = "same input") -> Contestant:
    return Contestant(
        name=name,
        test_case=LLMTestCase(input=input_, actual_output="output"),
    )


def test_empty_contestants_raises_value_error():
    with pytest.raises(ValueError, match="at least two contestants"):
        ArenaTestCase(contestants=[])


def test_single_contestant_raises_value_error():
    with pytest.raises(ValueError, match="at least two contestants"):
        ArenaTestCase(contestants=[_contestant("only")])


def test_two_contestants_with_same_input_are_accepted():
    arena = ArenaTestCase(
        contestants=[
            _contestant("version 1"),
            _contestant("version 2"),
        ]
    )
    assert len(arena.contestants) == 2


def test_different_inputs_still_rejected_after_count_check():
    with pytest.raises(ValueError, match="same 'input'"):
        ArenaTestCase(
            contestants=[
                _contestant("version 1", input_="input a"),
                _contestant("version 2", input_="input b"),
            ]
        )


def test_param_check_rejects_empty_contestants():
    # Bypass __post_init__ to exercise the metric-boundary guard directly.
    arena = ArenaTestCase.__new__(ArenaTestCase)
    arena.contestants = []
    arena.multimodal = False

    with pytest.raises(ValueError, match="at least two contestants"):
        check_arena_test_case_params(
            arena,
            [],
            DummyArenaMetric(),
        )


def test_param_check_rejects_single_contestant():
    arena = ArenaTestCase.__new__(ArenaTestCase)
    arena.contestants = [_contestant("only")]
    arena.multimodal = False

    with pytest.raises(ValueError, match="at least two contestants"):
        check_arena_test_case_params(
            arena,
            [],
            DummyArenaMetric(),
        )


def test_param_check_passes_for_valid_two_contestant_arena():
    arena = ArenaTestCase(
        contestants=[
            _contestant("version 1"),
            _contestant("version 2"),
        ]
    )
    # No exception: the guard must not reject valid arenas.
    check_arena_test_case_params(
        arena,
        [],
        DummyArenaMetric(),
    )


def test_param_check_still_raises_missing_params_error():
    # The pre-existing required-param validation must keep working: an arena
    # whose contestants lack a required parameter should raise the familiar
    # MissingTestCaseParamsError, not be masked by the count guard.
    arena = ArenaTestCase(
        contestants=[
            Contestant(
                name="version 1",
                test_case=LLMTestCase(input="input", actual_output=None),
            ),
            Contestant(
                name="version 2",
                test_case=LLMTestCase(input="input", actual_output="output"),
            ),
        ]
    )

    with pytest.raises(MissingTestCaseParamsError):
        check_arena_test_case_params(
            arena,
            [SingleTurnParams.ACTUAL_OUTPUT],
            DummyArenaMetric(),
        )
