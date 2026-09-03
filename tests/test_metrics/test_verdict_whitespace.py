"""Verdicts must survive surrounding whitespace.

Verdict strings come straight from a model's JSON and are stored on a plain
`str` pydantic field, which does not strip. A model that emits `"yes "` instead
of `"yes"` used to fail the comparison silently and drive the score to zero.

These drive the metrics through `measure()` with a stub model rather than
assigning to `metric.verdicts`, so the state under test is one the normal flow
produces.
"""

import json
from typing import Any, Optional

import pytest

from deepeval.metrics import ContextualRecallMetric, ContextualRelevancyMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase


class _StubModel(DeepEvalBaseLLM):
    def __init__(self, payloads: dict[str, Any], pad: bool) -> None:
        self.payloads = payloads
        self.pad = pad
        super().__init__()

    def load_model(self) -> "_StubModel":
        return self

    def get_model_name(self) -> str:
        return "stub"

    def _payload_for(self, schema: Any) -> dict[str, Any]:
        data = self.payloads[schema.__name__]
        if self.pad:
            # A trailing space is what a model actually emits; the point is that
            # nothing between the response and the score removes it.
            data = json.loads(json.dumps(data).replace('"yes"', '"yes "'))
        return data

    def generate(
        self, prompt: str, schema: Optional[Any] = None, **kwargs: Any
    ) -> Any:
        if schema is None:
            return json.dumps({"reason": "stub"})
        return schema(**self._payload_for(schema))

    async def a_generate(
        self, prompt: str, schema: Optional[Any] = None, **kwargs: Any
    ) -> Any:
        return self.generate(prompt, schema=schema, **kwargs)


_TEST_CASE = LLMTestCase(
    input="q",
    actual_output="a",
    expected_output="s1. s2. s3. s4.",
    retrieval_context=["c1", "c2"],
)

_RECALL_PAYLOADS = {
    "Verdicts": {
        "verdicts": [
            {"verdict": "yes", "reason": "supported"},
            {"verdict": "yes", "reason": "supported"},
            {"verdict": "no", "reason": "not supported"},
            {"verdict": "yes", "reason": "supported"},
        ]
    },
    "ContextualRecallScoreReason": {"reason": "stub"},
}

_RELEVANCY_PAYLOADS = {
    "ContextualRelevancyVerdicts": {
        "verdicts": [
            {"statement": "s1", "verdict": "yes", "reason": None},
            {"statement": "s2", "verdict": "yes", "reason": None},
            {"statement": "s3", "verdict": "no", "reason": "off topic"},
            {"statement": "s4", "verdict": "yes", "reason": None},
        ]
    },
    "ContextualRelevancyScoreReason": {"reason": "stub"},
}


@pytest.mark.parametrize("pad", [False, True], ids=["clean", "padded"])
def test_contextual_recall_ignores_whitespace_around_the_verdict(
    pad: bool,
) -> None:
    metric = ContextualRecallMetric(
        model=_StubModel(_RECALL_PAYLOADS, pad=pad),
        async_mode=False,
        include_reason=False,
    )
    metric.measure(_TEST_CASE, _show_indicator=False)

    assert metric.score == 0.75


@pytest.mark.parametrize("pad", [False, True], ids=["clean", "padded"])
def test_contextual_relevancy_ignores_whitespace_around_the_verdict(
    pad: bool,
) -> None:
    metric = ContextualRelevancyMetric(
        model=_StubModel(_RELEVANCY_PAYLOADS, pad=pad),
        async_mode=False,
        include_reason=False,
    )
    metric.measure(_TEST_CASE, _show_indicator=False)

    assert metric.score == 0.75
