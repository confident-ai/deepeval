"""Regression tests for verdict whitespace normalization.

Issue #3057: verdict-comparison sites compared ``verdict.verdict.lower()``
against "yes"/"no" without stripping surrounding whitespace. A model emitting
verdicts like ``"yes "`` (trailing space) failed the equality check, so
ContextualRecallMetric and ContextualRelevancyMetric scored 0 matched verdicts
out of 4 (0.0 instead of 0.75) and reason prompts mislabeled every padded
verdict as unsupportive/irrelevant.

These tests drive ``measure()`` end-to-end with a stubbed DeepEvalBaseLLM that
returns fixed verdict payloads and records the reason-generation prompts. They
make no network calls and do not require OPENAI_API_KEY.
"""

import json

import pytest

from deepeval.metrics import (
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    TurnContextualRecallMetric,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import ConversationalTestCase, LLMTestCase, Turn


RECALL_PAYLOADS = {
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

ALL_YES_RECALL_PAYLOADS = {
    "Verdicts": {
        "verdicts": [
            {"verdict": "yes", "reason": "supported"},
            {"verdict": "yes", "reason": "supported"},
            {"verdict": "yes", "reason": "supported"},
            {"verdict": "yes", "reason": "supported"},
        ]
    },
    "ContextualRecallScoreReason": {"reason": "stub"},
}

RELEVANCY_PAYLOADS = {
    "ContextualRelevancyVerdicts": {
        "verdicts": [
            {"statement": "s1", "verdict": "yes", "reason": "supported"},
            {"statement": "s2", "verdict": "yes", "reason": "supported"},
            {"statement": "s3", "verdict": "no", "reason": "not supported"},
            {"statement": "s4", "verdict": "yes", "reason": "supported"},
        ]
    },
    "ContextualRelevancyScoreReason": {"reason": "stub"},
}

ALL_YES_RELEVANCY_PAYLOADS = {
    "ContextualRelevancyVerdicts": {
        "verdicts": [
            {"statement": "s1", "verdict": "yes", "reason": "supported"},
            {"statement": "s2", "verdict": "yes", "reason": "supported"},
            {"statement": "s3", "verdict": "yes", "reason": "supported"},
            {"statement": "s4", "verdict": "yes", "reason": "supported"},
        ]
    },
    "ContextualRelevancyScoreReason": {"reason": "stub"},
}

RECALL_CASE = LLMTestCase(
    input="q",
    actual_output="a",
    expected_output="s1. s2. s3. s4.",
    retrieval_context=["c1", "c2"],
)

RELEVANCY_CASE = LLMTestCase(input="q", retrieval_context=["c1"])

TURN_CASE = ConversationalTestCase(
    turns=[
        Turn(role="user", content="q"),
        Turn(role="assistant", content="a", retrieval_context=["c1"]),
    ],
    expected_outcome="s1. s2. s3. s4.",
)


def _deepcopy(data):
    return json.loads(json.dumps(data))


def _map_verdicts(data, verdict_fn):
    data = _deepcopy(data)
    for verdict in data.get("verdicts", []):
        verdict["verdict"] = verdict_fn(verdict["verdict"])
    return data


def pad_yes_verdicts(data):
    return _map_verdicts(data, lambda v: v + " " if v == "yes" else v)


def pad_no_verdicts(data):
    return _map_verdicts(data, lambda v: v + " " if v == "no" else v)


def upper_yes_verdicts(data):
    return _map_verdicts(data, lambda v: v.upper() if v == "yes" else v)


def newline_yes_verdicts(data):
    return _map_verdicts(data, lambda v: v + "\n" if v == "yes" else v)


def tab_yes_verdicts(data):
    return _map_verdicts(data, lambda v: v + "\t" if v == "yes" else v)


def leading_yes_verdicts(data):
    return _map_verdicts(data, lambda v: " " + v if v == "yes" else v)


def whitespace_only_third_verdict(data):
    data = _deepcopy(data)
    if "verdicts" in data:
        data["verdicts"][2]["verdict"] = "  "
    return data


class VerdictStubModel(DeepEvalBaseLLM):
    """DeepEvalBaseLLM returning fixed payloads per schema.

    ``transform`` optionally rewrites the schema payload before it is
    returned, letting tests pad / uppercase / whitespace-ify verdict strings.
    When ``captured`` is set, prompts for ``capture_schema`` are recorded so
    tests can assert on the exact reason-generation prompt.
    """

    def __init__(
        self, payloads, transform=None, captured=None, capture_schema=None
    ):
        self.payloads = payloads
        self.transform = transform
        self.captured = captured
        self.capture_schema = capture_schema
        super().__init__()

    def load_model(self):
        return self

    def get_model_name(self):
        return "verdict-stub"

    def generate(self, prompt, schema=None, **kwargs):
        if schema is None:
            return json.dumps({"reason": "stub"})
        data = self.payloads[schema.__name__]
        if self.transform is not None:
            data = self.transform(data)
        if self.captured is not None and schema.__name__ == self.capture_schema:
            self.captured.append(prompt)
        return schema(**data)

    async def a_generate(self, prompt, schema=None, **kwargs):
        return self.generate(prompt, schema=schema, **kwargs)


def _measure_with_stub(
    metric_cls,
    payloads,
    case,
    *,
    transform=None,
    async_mode=False,
    include_reason=False,
    captured=None,
    capture_schema=None,
):
    model = VerdictStubModel(
        payloads,
        transform=transform,
        captured=captured,
        capture_schema=capture_schema,
    )
    metric = metric_cls(
        model=model, async_mode=async_mode, include_reason=include_reason
    )
    return metric.measure(case, _show_indicator=False)


def test_contextual_recall_score_baseline():
    assert (
        _measure_with_stub(ContextualRecallMetric, RECALL_PAYLOADS, RECALL_CASE)
        == 0.75
    )


def test_contextual_recall_score_with_padded_yes_verdicts():
    assert (
        _measure_with_stub(
            ContextualRecallMetric,
            RECALL_PAYLOADS,
            RECALL_CASE,
            transform=pad_yes_verdicts,
        )
        == 0.75
    )


def test_contextual_relevancy_score_baseline():
    assert (
        _measure_with_stub(
            ContextualRelevancyMetric, RELEVANCY_PAYLOADS, RELEVANCY_CASE
        )
        == 0.75
    )


def test_contextual_relevancy_score_with_padded_yes_verdicts():
    assert (
        _measure_with_stub(
            ContextualRelevancyMetric,
            RELEVANCY_PAYLOADS,
            RELEVANCY_CASE,
            transform=pad_yes_verdicts,
        )
        == 0.75
    )


def test_uppercase_verdicts_count_as_yes():
    for metric_cls, payloads, case in (
        (ContextualRecallMetric, RECALL_PAYLOADS, RECALL_CASE),
        (ContextualRelevancyMetric, RELEVANCY_PAYLOADS, RELEVANCY_CASE),
    ):
        assert (
            _measure_with_stub(
                metric_cls,
                payloads,
                case,
                transform=upper_yes_verdicts,
            )
            == 0.75
        )


def test_whitespace_only_verdict_does_not_count_as_yes():
    for metric_cls, payloads, case in (
        (ContextualRecallMetric, ALL_YES_RECALL_PAYLOADS, RECALL_CASE),
        (ContextualRelevancyMetric, ALL_YES_RELEVANCY_PAYLOADS, RELEVANCY_CASE),
    ):
        assert (
            _measure_with_stub(
                metric_cls,
                payloads,
                case,
                transform=whitespace_only_third_verdict,
            )
            == 0.75
        )


def test_newline_and_tab_padded_verdicts_count_as_yes():
    for transform in (newline_yes_verdicts, tab_yes_verdicts):
        assert (
            _measure_with_stub(
                ContextualRecallMetric,
                RECALL_PAYLOADS,
                RECALL_CASE,
                transform=transform,
            )
            == 0.75
        )


def test_leading_whitespace_verdicts_count_as_yes():
    assert (
        _measure_with_stub(
            ContextualRecallMetric,
            RECALL_PAYLOADS,
            RECALL_CASE,
            transform=leading_yes_verdicts,
        )
        == 0.75
    )


@pytest.mark.parametrize("async_mode", [False, True])
def test_contextual_recall_reason_prompt_with_padded_yes(async_mode):
    captured = []
    score = _measure_with_stub(
        ContextualRecallMetric,
        RECALL_PAYLOADS,
        RECALL_CASE,
        transform=pad_yes_verdicts,
        async_mode=async_mode,
        include_reason=True,
        captured=captured,
        capture_schema="ContextualRecallScoreReason",
    )
    assert score == 0.75
    assert len(captured) == 1
    prompt = captured[0]
    assert (
        "Supportive Reasons:\n['supported', 'supported', 'supported']" in prompt
    )
    assert "Unsupportive Reasons:\n['not supported']" in prompt


@pytest.mark.parametrize("async_mode", [False, True])
def test_contextual_relevancy_reason_prompt_with_padded_no(async_mode):
    captured = []
    score = _measure_with_stub(
        ContextualRelevancyMetric,
        RELEVANCY_PAYLOADS,
        RELEVANCY_CASE,
        transform=pad_no_verdicts,
        async_mode=async_mode,
        include_reason=True,
        captured=captured,
        capture_schema="ContextualRelevancyScoreReason",
    )
    assert score == 0.75
    assert len(captured) == 1
    prompt = captured[0]
    assert (
        "Reasons for why the retrieval context is irrelevant to the input:\n"
        "['not supported']"
    ) in prompt
    assert (
        "Statement in the retrieval context that is relevant to the input:\n"
        "['s1', 's2', 's4']"
    ) in prompt


@pytest.mark.parametrize("async_mode", [False, True])
def test_turn_contextual_recall_reason_prompt_with_padded_yes(async_mode):
    captured = []
    score = _measure_with_stub(
        TurnContextualRecallMetric,
        RECALL_PAYLOADS,
        TURN_CASE,
        transform=pad_yes_verdicts,
        async_mode=async_mode,
        include_reason=True,
        captured=captured,
        capture_schema="ContextualRecallScoreReason",
    )
    assert score == 0.75
    prompts = [p for p in captured if "Supportive Reasons:" in p]
    assert len(prompts) == 1
    prompt = prompts[0]
    assert (
        "Supportive Reasons:\n['supported', 'supported', 'supported']" in prompt
    )
    assert "Unsupportive Reasons:\n['not supported']" in prompt
