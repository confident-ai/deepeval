"""A reused metric instance must not inherit the previous measure's failure.

`prepare_measure` already zeroes the per-measure cost/token counters, but a
stale `error` from an earlier failed `measure()` survived every later call:
`is_successful()` short-circuits to False whenever `error is not None`, so a
metric that failed once kept reporting `success=False` even when a later
`measure()` recomputed a perfect score. The evaluate() executors reset both
fields between test cases (deepeval/evaluate/execute/agentic.py); the direct
measure()/a_measure() API now gets the same reset in `prepare_measure`.
"""

import pytest

from deepeval.metrics.answer_relevancy import AnswerRelevancyMetric
from deepeval.metrics.utils import prepare_measure
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase


class ScriptedModel(DeepEvalBaseLLM):
    """Custom (non-native) judge whose responses come from a scripted queue."""

    def __init__(self, responses):
        self.responses = list(responses)

    def load_model(self):
        return None

    def generate(self, prompt: str, schema=None, **kwargs):
        out = self.responses.pop(0)
        return out(prompt) if callable(out) else out

    async def a_generate(self, prompt: str, schema=None, **kwargs):
        return self.generate(prompt, schema=schema, **kwargs)

    def get_model_name(self):
        return "scripted-stub"


GOOD_STATEMENTS = '{"statements": ["Paris is in France."]}'
GOOD_VERDICTS = '{"verdicts": [{"verdict": "yes", "reason": "r", "statement": "Paris is in France."}]}'
GOOD_REASON = '{"reason": "relevant"}'

TEST_CASE = LLMTestCase(
    input="Where is Paris?", actual_output="Paris is in France."
)


def test_recovered_measure_reports_success_after_earlier_failure():
    """Second measure() on the same instance: perfect verdicts, so success=True."""
    model = ScriptedModel(
        [
            lambda p: "NOT JSON AT ALL",  # statements call fails
            lambda p: "x",  # reason probe (unused path)
        ]
    )
    metric = AnswerRelevancyMetric(
        model=model, async_mode=False, include_reason=True
    )

    with pytest.raises(ValueError):
        metric.measure(TEST_CASE)
    assert metric.error is not None, "first measure must record its error"

    model.responses = [
        lambda p: GOOD_STATEMENTS,
        lambda p: GOOD_VERDICTS,
        lambda p: GOOD_REASON,
    ]
    score = metric.measure(TEST_CASE)

    assert score == 1.0
    assert (
        metric.error is None
    ), "a recovered measure must clear the stale error"
    assert (
        metric.is_successful() is True
    ), "a reused metric must not report failure for a measure that succeeded"


def test_prepare_measure_clears_stale_error_and_skipped():
    """Unit-level: prepare_measure resets the per-measure outcome fields."""
    metric = AnswerRelevancyMetric(
        model=ScriptedModel([]), async_mode=False, include_reason=False
    )
    metric.error = "stale error from a previous run"
    metric.skipped = True

    prepare_measure(metric, TEST_CASE)

    assert metric.error is None
    assert metric.skipped is False


def test_validation_errors_are_still_recorded():
    """A genuinely invalid test case still sets error and raises — the reset
    happens after validation, so it cannot mask new failures."""
    metric = AnswerRelevancyMetric(
        model=ScriptedModel([]), async_mode=False, include_reason=False
    )
    broken = LLMTestCase(input="Where is Paris?", actual_output="")

    with pytest.raises(Exception):
        metric.measure(broken)

    assert metric.error is not None, "validation errors must still be recorded"
