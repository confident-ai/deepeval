import pytest
from typing import Optional

from deepeval.test_run.api import (
    ConversationalApiTestCase,
    LLMApiTestCase,
)
from deepeval.test_run.test_run import TestRun as DeepEvalTestRun


def _llm_case(
    name: str, success: bool, evaluation_cost: Optional[float] = None
) -> LLMApiTestCase:
    return LLMApiTestCase(
        name=name,
        input="question",
        success=success,
        evaluationCost=evaluation_cost,
    )


def _conversational_case(
    name: str, success: bool, evaluation_cost: Optional[float] = None
) -> ConversationalApiTestCase:
    return ConversationalApiTestCase(
        name=name,
        success=success,
        metricsData=[],
        evaluationCost=evaluation_cost,
    )


def test_add_test_case_replaces_retried_llm_test_case_by_name():
    test_run = DeepEvalTestRun(testFile="test_retry.py")

    test_run.add_test_case(
        _llm_case("test_flaky", success=False, evaluation_cost=0.01)
    )
    test_run.add_test_case(
        _llm_case("test_flaky", success=True, evaluation_cost=0.02)
    )

    assert len(test_run.test_cases) == 1
    assert test_run.test_cases[0].success is True
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
