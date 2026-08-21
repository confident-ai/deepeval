import pytest
from typing import Optional

from deepeval.test_run.api import (
    ConversationalApiTestCase,
    LLMApiTestCase,
)
from deepeval.test_run.test_run import TestRun as DeepEvalTestRun


def _llm_case(
    name: str,
    success: bool,
    evaluation_cost: Optional[float] = None,
    input: str = "question",
    actual_output: Optional[str] = None,
    expected_output: Optional[str] = None,
) -> LLMApiTestCase:
    return LLMApiTestCase(
        name=name,
        input=input,
        actualOutput=actual_output,
        expectedOutput=expected_output,
        success=success,
        evaluationCost=evaluation_cost,
    )


def _conversational_case(
    name: str,
    success: bool,
    evaluation_cost: Optional[float] = None,
    scenario: Optional[str] = None,
) -> ConversationalApiTestCase:
    return ConversationalApiTestCase(
        name=name,
        success=success,
        metricsData=[],
        evaluationCost=evaluation_cost,
        scenario=scenario,
    )


def test_add_test_case_replaces_retried_llm_test_case_by_name():
    test_run = DeepEvalTestRun(testFile="test_retry.py")

    test_run.add_test_case(
        _llm_case(
            "test_flaky",
            success=False,
            evaluation_cost=0.01,
            actual_output="failed attempt",
        )
    )
    test_run.add_test_case(
        _llm_case(
            "test_flaky",
            success=True,
            evaluation_cost=0.02,
            actual_output="passing attempt",
        )
    )

    assert len(test_run.test_cases) == 1
    assert test_run.test_cases[0].success is True
    assert test_run.test_cases[0].actual_output == "passing attempt"
    assert test_run.evaluation_cost == pytest.approx(0.02)


def test_add_test_case_keeps_distinct_llm_test_case_names():
    test_run = DeepEvalTestRun(testFile="test_retry.py")

    test_run.add_test_case(
        _llm_case("test_one", success=True, evaluation_cost=0.01)
    )
    test_run.add_test_case(
        _llm_case("test_two", success=False, evaluation_cost=0.03)
    )

    assert [case.name for case in test_run.test_cases] == [
        "test_one",
        "test_two",
    ]
    assert test_run.evaluation_cost == pytest.approx(0.04)


def test_add_test_case_keeps_same_name_llm_test_cases_with_different_inputs():
    test_run = DeepEvalTestRun(testFile="test_retry.py")

    test_run.add_test_case(
        _llm_case(
            "test_parametrized", success=True, evaluation_cost=0.01, input="a"
        )
    )
    test_run.add_test_case(
        _llm_case(
            "test_parametrized",
            success=False,
            evaluation_cost=0.03,
            input="b",
        )
    )

    assert [case.input for case in test_run.test_cases] == ["a", "b"]
    assert test_run.evaluation_cost == pytest.approx(0.04)


def test_add_test_case_keeps_same_name_llm_test_cases_with_different_expected_outputs():
    test_run = DeepEvalTestRun(testFile="test_retry.py")

    test_run.add_test_case(
        _llm_case(
            "test_parametrized",
            success=True,
            evaluation_cost=0.01,
            expected_output="answer a",
        )
    )
    test_run.add_test_case(
        _llm_case(
            "test_parametrized",
            success=False,
            evaluation_cost=0.03,
            expected_output="answer b",
        )
    )

    assert [case.expected_output for case in test_run.test_cases] == [
        "answer a",
        "answer b",
    ]
    assert test_run.evaluation_cost == pytest.approx(0.04)


def test_add_test_case_clears_retried_llm_cost_when_latest_cost_is_none():
    test_run = DeepEvalTestRun(testFile="test_retry.py")

    test_run.add_test_case(
        _llm_case("test_flaky", success=False, evaluation_cost=0.01)
    )
    test_run.add_test_case(
        _llm_case("test_flaky", success=True, evaluation_cost=None)
    )

    assert len(test_run.test_cases) == 1
    assert test_run.evaluation_cost == pytest.approx(0.0)


def test_add_test_case_replaces_retried_conversational_test_case_by_name():
    test_run = DeepEvalTestRun(testFile="test_retry.py")

    test_run.add_test_case(
        _conversational_case(
            "test_conversation", success=False, evaluation_cost=0.04
        )
    )
    test_run.add_test_case(
        _conversational_case(
            "test_conversation", success=True, evaluation_cost=0.05
        )
    )

    assert len(test_run.conversational_test_cases) == 1
    assert test_run.conversational_test_cases[0].success is True
    assert test_run.evaluation_cost == pytest.approx(0.05)


def test_add_test_case_keeps_same_name_conversations_with_different_scenarios():
    test_run = DeepEvalTestRun(testFile="test_retry.py")

    test_run.add_test_case(
        _conversational_case(
            "test_conversation",
            success=True,
            evaluation_cost=0.04,
            scenario="scenario one",
        )
    )
    test_run.add_test_case(
        _conversational_case(
            "test_conversation",
            success=False,
            evaluation_cost=0.05,
            scenario="scenario two",
        )
    )

    assert [case.scenario for case in test_run.conversational_test_cases] == [
        "scenario one",
        "scenario two",
    ]
    assert test_run.evaluation_cost == pytest.approx(0.09)
