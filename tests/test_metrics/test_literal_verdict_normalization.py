"""Regression tests for Literal verdict schemas on the loose-JSON path.

`CitationFaithfulnessVerdict` and `RelevancyVerdict` type their verdict as a
`Literal`, and a non-native judge's reply is built straight into the schema.
A judge that answered "Faithful" or "tp" used to crash `measure()` with a
pydantic `ValidationError` even though the verdict is unambiguous. Case and
surrounding whitespace are now folded before validation, while a verdict that
is genuinely outside the vocabulary still raises.

These tests use a stub judge that returns raw JSON strings, so they need no
LLM provider API key.
"""

import json

import pytest
from pydantic import ValidationError

from deepeval.metrics import TopicAdherenceMetric
from deepeval.metrics.community import CitationFaithfulnessMetric
from deepeval.metrics.community.citation_faithfulness.schema import (
    CitationFaithfulnessVerdict,
)
from deepeval.metrics.topic_adherence.schema import RelevancyVerdict
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import ConversationalTestCase, LLMTestCase, Turn


class _LooseJudge(DeepEvalBaseLLM):
    """Non-native judge that replies with a fixed verdict as raw JSON."""

    def __init__(self, verdict: str):
        self.verdict = verdict
        super().__init__(model="loose-judge")

    def load_model(self, *args, **kwargs):
        return None

    def generate(self, prompt, *args, schema=None, **kwargs) -> str:
        name = schema.__name__ if schema is not None else ""
        if name == "QAPairs":
            return json.dumps(
                {
                    "qa_pairs": [
                        {
                            "question": "Tell me about thermal cameras.",
                            "response": "Thermal cameras detect infrared radiation.",
                        }
                    ]
                }
            )
        if name == "RelevancyVerdict":
            return json.dumps({"verdict": self.verdict, "reason": "stub"})
        if name == "TopicAdherenceReason":
            return json.dumps({"reason": "stub"})
        return json.dumps({"verdict": self.verdict, "reasoning": "stub"})

    async def a_generate(self, prompt, *args, schema=None, **kwargs) -> str:
        return self.generate(prompt, schema=schema, **kwargs)

    def get_model_name(self, *args, **kwargs) -> str:
        return "loose-judge"


def _citation_case() -> LLMTestCase:
    return LLMTestCase(
        input="Who wrote Hamlet?",
        actual_output="Hamlet was written by Shakespeare. [1]",
        retrieval_context=["Hamlet is a play by William Shakespeare."],
    )


def _topic_case() -> ConversationalTestCase:
    return ConversationalTestCase(
        turns=[
            Turn(role="user", content="Tell me about thermal cameras."),
            Turn(
                role="assistant",
                content="Thermal cameras detect infrared radiation.",
            ),
        ]
    )


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize(
    "raw, expected_score",
    [
        ("Faithful", 1.0),
        ("FAITHFUL", 1.0),
        (" faithful\n", 1.0),
        ("Unfaithful", 0.0),
        (" UNFAITHFUL ", 0.0),
    ],
)
def test_citation_faithfulness_scores_case_variant_verdicts(
    raw, expected_score, async_mode
):
    metric = CitationFaithfulnessMetric(
        model=_LooseJudge(raw), async_mode=async_mode
    )

    metric.measure(_citation_case(), _show_indicator=False)

    assert metric.score == expected_score
    assert metric.verdict.verdict in ("faithful", "unfaithful")


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize(
    "raw, expected_score",
    [
        ("tp", 1.0),
        ("Tn", 1.0),
        (" fp ", 0.0),
        ("fn\n", 0.0),
    ],
)
def test_topic_adherence_scores_case_variant_verdicts(
    raw, expected_score, async_mode
):
    metric = TopicAdherenceMetric(
        relevant_topics=["thermal imaging"],
        model=_LooseJudge(raw),
        async_mode=async_mode,
    )

    metric.measure(_topic_case(), _show_indicator=False)

    assert metric.score == expected_score


def test_out_of_vocabulary_verdicts_still_raise():
    with pytest.raises(ValidationError):
        CitationFaithfulnessVerdict(verdict="partially faithful")
    with pytest.raises(ValidationError):
        RelevancyVerdict(verdict="maybe", reason="stub")
