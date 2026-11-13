"""Regression tests: QAG verdicts wrapped in markdown or quotes.

A judge that answers with loose JSON (a custom model whose ``generate``
returns a string) may write ``"**yes**"``, ``"`no`"`` or ``'"no"'``. These
were dropped as out-of-vocabulary, which changed the score: when every
verdict was wrapped, the metric fell back to the empty-list score of 1.
"""

import json

import pytest

from deepeval.metrics import FaithfulnessMetric
from deepeval.metrics.base_metric import Verdict, YES_NO_BORDERLINE
from deepeval.metrics.utils import normalize_qag_verdict
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("**yes**", Verdict.YES),
        ("**No**", Verdict.NO),
        ("`no`", Verdict.NO),
        ('"yes"', Verdict.YES),
        ("'no'", Verdict.NO),
        ("\u201cno\u201d", Verdict.NO),
        ("[no]", Verdict.NO),
        ("\u200byes", Verdict.YES),
        ("**Yes**, the claim is supported.", Verdict.YES),
        ("**idk**", Verdict.BORDERLINE),
    ],
)
def test_wrapped_verdict_is_read(raw, expected):
    assert normalize_qag_verdict(raw, YES_NO_BORDERLINE) == expected


@pytest.mark.parametrize(
    "raw", ["**maybe**", "`yesterday`", "**", "", "1. yes"]
)
def test_wrapped_unknown_word_is_still_dropped(raw):
    assert normalize_qag_verdict(raw, YES_NO_BORDERLINE) is None


class _JsonJudge(DeepEvalBaseLLM):
    """Custom model that returns loose JSON strings instead of schema objects."""

    def __init__(self, verdicts):
        self.replies = {
            "Truths": {"truths": ["The Eiffel Tower is in Paris."]},
            "Claims": {"claims": ["It is in Berlin.", "It was built in 1999."]},
            "Verdicts": {
                "verdicts": [{"verdict": v, "reason": "r"} for v in verdicts]
            },
            "FaithfulnessScoreReason": {"reason": "r"},
        }
        super().__init__("json-judge")

    def load_model(self):
        return None

    def generate(self, prompt, schema=None):
        return json.dumps(self.replies[schema.__name__])

    async def a_generate(self, prompt, schema=None):
        return self.generate(prompt, schema=schema)

    def get_model_name(self):
        return "json-judge"


TEST_CASE = LLMTestCase(
    input="Where is the Eiffel Tower?",
    actual_output="It is in Berlin. It was built in 1999.",
    retrieval_context=["The Eiffel Tower is in Paris."],
)


@pytest.mark.parametrize(
    "verdicts, expected",
    [
        (["**no**", "**no**"], 0.0),
        (["`no`", '"no"'], 0.0),
        (["**yes**", "no"], 0.5),
    ],
)
@pytest.mark.parametrize("async_mode", [False, True])
def test_faithfulness_scores_wrapped_verdicts(verdicts, expected, async_mode):
    metric = FaithfulnessMetric(
        model=_JsonJudge(verdicts), async_mode=async_mode
    )
    metric.measure(TEST_CASE, _show_indicator=False)
    assert len(metric.verdicts) == len(verdicts)
    assert metric.score == expected
