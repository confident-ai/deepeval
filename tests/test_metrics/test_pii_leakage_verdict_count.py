"""Regression tests for PIILeakageMetric's score denominator.

The verdicts template asks the judge for exactly one verdict per extracted
PII item, but the judge can return a truncated list, or reply with
out-of-vocabulary verdicts that ``_verdicts_from_json`` drops. Previously
``score_qag_verdicts`` divided by the number of usable verdicts, so every
missing verdict shrank the denominator and inflated the score: a judge that
assessed 1 of 3 PII items produced a perfect privacy score (issue #3346).

These tests use a stub model returning fixed responses, so they run without
an LLM provider API key. They cover sync and async execution, and both
schema (structured) and raw JSON judge responses.
"""

import json

import pytest

from deepeval.metrics import PIILeakageMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase

TEST_CASE = LLMTestCase(
    input="How do I contact billing?",
    actual_output="Email alice@example.com, call 555-1234.",
)

THREE_PII = ["alice@example.com", "555-1234", "4242-4242-4242-4242"]


class _FixedPIIJudge(DeepEvalBaseLLM):
    """Minimal DeepEvalBaseLLM with fixed extraction and verdicts."""

    def __init__(self, verdicts, structured: bool):
        self.verdicts = verdicts
        self.structured = structured
        super().__init__("fixed-pii-judge")

    def load_model(self, *args, **kwargs):
        return None

    def generate(self, prompt, schema=None):
        if schema is not None and "extracted_pii" in schema.model_fields:
            data = {"extracted_pii": THREE_PII}
        elif schema is not None and "verdicts" in schema.model_fields:
            data = {"verdicts": self.verdicts}
        else:
            data = {"reason": "Fixed reason."}
        if self.structured and schema is not None:
            return schema(**data)
        return json.dumps(data)

    async def a_generate(self, prompt, schema=None):
        return self.generate(prompt, schema)

    def get_model_name(self, *args, **kwargs) -> str:
        return "fixed-pii-judge"


def _measure(model, run_async: bool) -> float:
    metric = PIILeakageMetric(
        model=model,
        async_mode=run_async,
        include_reason=False,
        threshold=0.5,
    )
    score = metric.measure(TEST_CASE, _show_indicator=False)
    return metric, score


@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("run_async", [False, True])
def test_truncated_verdict_list_must_not_score_perfect(structured, run_async):
    """Judge assesses only the first of 3 PII: score is 1/3, not 1."""
    model = _FixedPIIJudge([{"verdict": "no", "reason": "masked"}], structured)
    metric, score = _measure(model, run_async)
    assert score == pytest.approx(1 / 3)
    assert not metric.is_successful()


@pytest.mark.parametrize("run_async", [False, True])
def test_dropped_out_of_vocab_verdicts_must_not_score_perfect(run_async):
    """Raw-JSON judge: 2 of 3 verdicts out-of-vocabulary, dropped.

    Only the raw JSON path drops out-of-vocabulary verdicts; the structured
    path rejects them at schema validation, so this test is not parametrized
    over ``structured``.
    """
    model = _FixedPIIJudge(
        [
            {"verdict": "no", "reason": "masked"},
            {"verdict": "probably yes", "reason": "hmm"},
            {"verdict": "unclear", "reason": "bad scan"},
        ],
        structured=False,
    )
    metric, score = _measure(model, run_async)
    assert score == pytest.approx(1 / 3)
    assert not metric.is_successful()


@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("run_async", [False, True])
def test_complete_verdict_list_scores_unchanged(structured, run_async):
    """Control: all 3 PII assessed, none leak: score stays 1."""
    model = _FixedPIIJudge(
        [{"verdict": "no", "reason": "masked"}] * 3, structured
    )
    metric, score = _measure(model, run_async)
    assert score == pytest.approx(1.0)
    assert metric.is_successful()


@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("run_async", [False, True])
def test_detected_leak_still_fails(structured, run_async):
    """Polarity control: all 3 verdicts say leak: score is 0."""
    model = _FixedPIIJudge(
        [{"verdict": "yes", "reason": "leaked"}] * 3, structured
    )
    metric, score = _measure(model, run_async)
    assert score == pytest.approx(0.0)
    assert not metric.is_successful()
