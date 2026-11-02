"""Regression tests: verdict comparisons must ignore surrounding whitespace.

Three metrics compared the raw verdict string (``verdict.verdict.lower()``)
before scoring, so a model that emitted ``"yes "`` instead of ``"yes"`` was
counted the wrong way and the score collapsed (e.g. 0.75 -> 0.0) instead of
drifting. Every other verdict comparison in the repo already called
``.strip().lower()``; these were the outliers.

Drives the full ``measure()`` flow with a stub model so the verdicts take the
same path a real run does, rather than assigning to ``metric.verdicts``.
"""

import json

from deepeval.metrics import ContextualRecallMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase

VERDICTS_PAYLOAD = {
    "verdicts": [
        {"verdict": "yes", "reason": "supported"},
        {"verdict": "yes", "reason": "supported"},
        {"verdict": "no", "reason": "not supported"},
        {"verdict": "yes", "reason": "supported"},
    ]
}

REASON_PAYLOAD = {"reason": "stub"}


class StubModel(DeepEvalBaseLLM):
    """Returns a canned verdicts payload, optionally padding every "yes" with a
    trailing space — the exact whitespace that tripped the old comparisons."""

    def __init__(self, pad: bool):
        self.pad = pad
        super().__init__()

    def load_model(self):
        return self

    def get_model_name(self):
        return "stub"

    def generate(self, prompt, schema=None, **kwargs):
        if schema is None:
            return json.dumps(REASON_PAYLOAD)
        if schema.__name__ == "Verdicts":
            data = VERDICTS_PAYLOAD
            if self.pad:
                data = {
                    "verdicts": [
                        {**v, "verdict": "yes "} if v["verdict"] == "yes" else v
                        for v in VERDICTS_PAYLOAD["verdicts"]
                    ]
                }
            return schema(**data)
        return schema(**REASON_PAYLOAD)

    async def a_generate(self, prompt, schema=None, **kwargs):
        return self.generate(prompt, schema=schema, **kwargs)


def measure_contextual_recall(pad: bool) -> float:
    case = LLMTestCase(
        input="q",
        actual_output="a",
        expected_output="s1. s2. s3. s4.",
        retrieval_context=["c1", "c2"],
    )
    metric = ContextualRecallMetric(
        model=StubModel(pad=pad),
        async_mode=False,
        include_reason=False,
    )
    metric.measure(case, _show_indicator=False)
    return metric.score


def test_contextual_recall_unpadded_verdicts_score_half():
    # 3 of 4 verdicts are "yes" -> 0.75 regardless of padding.
    assert measure_contextual_recall(pad=False) == 0.75


def test_contextual_recall_padded_verdicts_score_half():
    # Regression: "yes " used to drop the score to 0.0 because the comparison
    # did not strip whitespace before checking against "yes".
    assert measure_contextual_recall(pad=True) == 0.75
