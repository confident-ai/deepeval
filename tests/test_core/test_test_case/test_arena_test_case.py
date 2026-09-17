from unittest.mock import MagicMock
import pytest

from deepeval.metrics.utils import check_arena_test_case_params
from deepeval.test_case import ArenaTestCase, LLMTestCase, SingleTurnParams
from deepeval.test_case.arena_test_case import Contestant


def test_arena_test_case_empty_contestants_raises_value_error():
    with pytest.raises(
        ValueError,
        match="An arena test case must have at least two contestants.",
    ):
        ArenaTestCase(contestants=[])


def test_arena_test_case_single_contestant_raises_value_error():
    contestant = Contestant(
        name="Model A",
        test_case=LLMTestCase(input="Hello", actual_output="Hi there"),
    )
    with pytest.raises(
        ValueError,
        match="An arena test case must have at least two contestants.",
    ):
        ArenaTestCase(contestants=[contestant])


def test_arena_test_case_duplicate_contestant_names_raises_value_error():
    c1 = Contestant(
        name="Model A",
        test_case=LLMTestCase(input="Hello", actual_output="Hi"),
    )
    c2 = Contestant(
        name="Model A",
        test_case=LLMTestCase(input="Hello", actual_output="Hey"),
    )
    with pytest.raises(
        ValueError, match="All contestant names must be unique."
    ):
        ArenaTestCase(contestants=[c1, c2])


def test_arena_test_case_mismatched_input_raises_value_error():
    c1 = Contestant(
        name="Model A",
        test_case=LLMTestCase(input="Hello", actual_output="Hi"),
    )
    c2 = Contestant(
        name="Model B",
        test_case=LLMTestCase(input="Different question", actual_output="Hey"),
    )
    with pytest.raises(
        ValueError, match="All contestants must have the same 'input'."
    ):
        ArenaTestCase(contestants=[c1, c2])


def test_arena_test_case_mismatched_expected_output_raises_value_error():
    c1 = Contestant(
        name="Model A",
        test_case=LLMTestCase(
            input="Hello", actual_output="Hi", expected_output="Expected 1"
        ),
    )
    c2 = Contestant(
        name="Model B",
        test_case=LLMTestCase(
            input="Hello", actual_output="Hey", expected_output="Expected 2"
        ),
    )
    with pytest.raises(
        ValueError,
        match="All contestants must have the same 'expected_output'.",
    ):
        ArenaTestCase(contestants=[c1, c2])


def test_arena_test_case_valid_initialization():
    c1 = Contestant(
        name="Model A",
        test_case=LLMTestCase(input="Hello", actual_output="Hi"),
    )
    c2 = Contestant(
        name="Model B",
        test_case=LLMTestCase(input="Hello", actual_output="Hey"),
    )
    arena_tc = ArenaTestCase(contestants=[c1, c2])
    assert len(arena_tc.contestants) == 2
    assert arena_tc.multimodal is False


def test_check_arena_test_case_params_validates_contestants_count():
    mock_metric = MagicMock()
    mock_metric.__name__ = "ArenaGEval"

    # Create a dummy object or bypass __post_init__ to test check_arena_test_case_params guard directly
    invalid_arena_tc = object.__new__(ArenaTestCase)
    invalid_arena_tc.contestants = []

    with pytest.raises(
        ValueError,
        match="An arena test case must have at least two contestants.",
    ):
        check_arena_test_case_params(
            arena_test_case=invalid_arena_tc,
            test_case_params=[SingleTurnParams.INPUT],
            metric=mock_metric,
        )
