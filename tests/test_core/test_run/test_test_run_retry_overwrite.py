"""Tests for the pytest.mark.flaky / pytest-rerunfailures retry-overwrite fix.

When assert_test() is retried, each attempt calls add_test_case(); the
implementation replaces an existing row with the same name so the
dashboard shows one row per logical test and evaluation_cost is not
double-counted across retries.

Covers:
- LLM: same name twice → one row, latest success wins
- LLM: two different names → two rows (unchanged)
- LLM: cost not double-counted on replace
- LLM: retry with None cost backs out previous cost
- LLM: first attempt None cost, retry sets cost
- LLM: three retries → one row, only last cost counted
- LLM: mixed retried + unique test → correct total cost
- Conversational: same name twice → one row
- Conversational: cost not double-counted
- Conversational: two different names → two rows
"""

from __future__ import annotations

from typing import Optional

import pytest

from deepeval.test_run.api import LLMApiTestCase, ConversationalApiTestCase
from deepeval.test_run.test_run import TestRun

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _llm(
    name: str, success: bool, cost: Optional[float] = None
) -> LLMApiTestCase:
    # LLMApiTestCase uses pydantic Field(alias=...) for several fields.
    # Without populate_by_name=True in the model config, aliases must be
    # used at construction time — not the Python attribute names.
    return LLMApiTestCase(
        name=name,
        input="q",
        success=success,
        evaluationCost=cost,
    )


def _conv(
    name: str, success: bool, cost: Optional[float] = None
) -> ConversationalApiTestCase:
    # metricsData has no default and must be supplied via its alias.
    return ConversationalApiTestCase(
        name=name,
        success=success,
        metricsData=[],
        evaluationCost=cost,
    )


def _fresh_run() -> TestRun:
    """Return a TestRun with empty lists and no accumulated cost."""
    # test_file uses alias testFile when populate_by_name is not enabled.
    return TestRun(testFile="test_file.py")


# ---------------------------------------------------------------------------
# LLM test cases
# ---------------------------------------------------------------------------


class TestLLMRetryOverwrite:
    def test_retry_produces_single_row(self):
        """Second call with the same name replaces the first row."""
        run = _fresh_run()
        run.add_test_case(_llm("test_foo", success=False))
        run.add_test_case(_llm("test_foo", success=True))

        assert len(run.test_cases) == 1
        assert run.test_cases[0].success is True

    def test_distinct_names_produce_two_rows(self):
        """Two different names → two rows (existing behaviour unchanged)."""
        run = _fresh_run()
        run.add_test_case(_llm("test_foo", success=True))
        run.add_test_case(_llm("test_bar", success=False))

        assert len(run.test_cases) == 2

    def test_retry_does_not_double_count_cost(self):
        """Run-level cost reflects only the *last* attempt's cost."""
        run = _fresh_run()
        run.add_test_case(_llm("test_foo", success=False, cost=0.01))
        run.add_test_case(_llm("test_foo", success=True, cost=0.02))

        assert run.evaluation_cost == pytest.approx(0.02)

    def test_retry_latest_cost_none_clears_previous(self):
        """If the retry reports no cost, the previous cost is backed out."""
        run = _fresh_run()
        run.add_test_case(_llm("test_foo", success=False, cost=0.01))
        run.add_test_case(_llm("test_foo", success=True, cost=None))

        # Previous 0.01 is subtracted; new attempt adds nothing.
        assert run.evaluation_cost == pytest.approx(0.0)

    def test_first_attempt_cost_none_retry_sets_cost(self):
        """First attempt has no cost; retry introduces a cost → cost is set."""
        run = _fresh_run()
        run.add_test_case(_llm("test_foo", success=False, cost=None))
        run.add_test_case(_llm("test_foo", success=True, cost=0.05))

        assert run.evaluation_cost == pytest.approx(0.05)

    def test_multiple_retries_keep_only_last(self):
        """Three attempts with the same name → one row, last success wins."""
        run = _fresh_run()
        run.add_test_case(_llm("test_foo", success=False, cost=0.01))
        run.add_test_case(_llm("test_foo", success=False, cost=0.02))
        run.add_test_case(_llm("test_foo", success=True, cost=0.03))

        assert len(run.test_cases) == 1
        assert run.test_cases[0].success is True
        assert run.evaluation_cost == pytest.approx(0.03)

    def test_cost_accumulates_correctly_for_distinct_tests_with_retry(self):
        """Mixed scenario: one retried test + one unique test → correct total."""
        run = _fresh_run()
        run.add_test_case(_llm("test_a", success=False, cost=0.01))  # attempt 1
        run.add_test_case(_llm("test_b", success=True, cost=0.05))  # unique
        run.add_test_case(
            _llm("test_a", success=True, cost=0.02)
        )  # attempt 2 (retry)

        assert len(run.test_cases) == 2
        assert run.evaluation_cost == pytest.approx(0.07)  # 0.02 + 0.05


# ---------------------------------------------------------------------------
# Conversational test cases
# ---------------------------------------------------------------------------


class TestConversationalRetryOverwrite:
    def test_retry_produces_single_row(self):
        """Second call with the same name replaces the first row."""
        run = _fresh_run()
        run.add_test_case(_conv("conv_test", success=False))
        run.add_test_case(_conv("conv_test", success=True))

        assert len(run.conversational_test_cases) == 1
        assert run.conversational_test_cases[0].success is True

    def test_retry_does_not_double_count_cost(self):
        run = _fresh_run()
        run.add_test_case(_conv("conv_test", success=False, cost=0.01))
        run.add_test_case(_conv("conv_test", success=True, cost=0.02))

        assert run.evaluation_cost == pytest.approx(0.02)

    def test_distinct_names_produce_two_rows(self):
        run = _fresh_run()
        run.add_test_case(_conv("conv_a", success=True))
        run.add_test_case(_conv("conv_b", success=False))

        assert len(run.conversational_test_cases) == 2
