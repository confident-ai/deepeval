from __future__ import annotations

import sys
import os

import pytest

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "examples",
        "retail_support_evaluation",
    ),
)

from retail_support import (
    RETAIL_SUPPORT_DATASET,
    POLICY_CORRECTNESS,
    FAITHFULNESS,
    ANSWER_RELEVANCY,
    SupportSample,
    build_test_cases,
    check_forbidden_patterns,
)

from deepeval import assert_test
from deepeval.test_case import LLMTestCase


@pytest.fixture(scope="module")
def all_samples() -> list[SupportSample]:
    return RETAIL_SUPPORT_DATASET


@pytest.fixture(scope="module")
def all_test_cases() -> list[LLMTestCase]:
    return build_test_cases(RETAIL_SUPPORT_DATASET)


class TestForbiddenCommitments:

    @pytest.mark.parametrize(
        "sample",
        [s for s in RETAIL_SUPPORT_DATASET if s.forbidden_patterns],
        ids=[s.case_id for s in RETAIL_SUPPORT_DATASET if s.forbidden_patterns],
    )
    def test_no_forbidden_pattern(self, sample: SupportSample) -> None:
        result = check_forbidden_patterns(sample)
        assert result["passed"], (
            f"[{sample.case_id}] Response contains forbidden pattern(s): "
            f"{result['matched']}\n\n"
            f"Response: {sample.llm_response}"
        )

    @pytest.mark.parametrize(
        "sample",
        [s for s in RETAIL_SUPPORT_DATASET if not s.forbidden_patterns],
        ids=[
            s.case_id
            for s in RETAIL_SUPPORT_DATASET
            if not s.forbidden_patterns
        ],
    )
    def test_positive_cases_no_forbidden_patterns_defined(
        self, sample: SupportSample
    ) -> None:
        result = check_forbidden_patterns(sample)
        assert result[
            "passed"
        ], f"[{sample.case_id}] Unexpected forbidden-pattern failure."


class TestGroundingAndRelevance:

    @pytest.mark.parametrize(
        "test_case",
        build_test_cases(RETAIL_SUPPORT_DATASET),
        ids=[s.case_id for s in RETAIL_SUPPORT_DATASET],
    )
    def test_faithfulness_and_relevancy(self, test_case: LLMTestCase) -> None:
        assert_test(test_case, [FAITHFULNESS, ANSWER_RELEVANCY])


class TestPolicyCorrectness:

    @pytest.mark.parametrize(
        "test_case",
        build_test_cases(RETAIL_SUPPORT_DATASET),
        ids=[s.case_id for s in RETAIL_SUPPORT_DATASET],
    )
    def test_policy_correctness(self, test_case: LLMTestCase) -> None:
        assert_test(test_case, [POLICY_CORRECTNESS])


CATEGORIES = sorted({s.category for s in RETAIL_SUPPORT_DATASET})


class TestByCategory:

    @pytest.mark.parametrize("category", CATEGORIES)
    def test_category(self, category: str) -> None:
        category_cases = [
            tc
            for tc, s in zip(
                build_test_cases(RETAIL_SUPPORT_DATASET), RETAIL_SUPPORT_DATASET
            )
            if s.category == category
        ]
        assert category_cases, f"No test cases for category '{category}'"
        for tc in category_cases:
            assert_test(
                tc, [FAITHFULNESS, ANSWER_RELEVANCY, POLICY_CORRECTNESS]
            )
