"""Tests for NamedReferenceAttributionMetric.

These tests use a fake DeepEvalBaseLLM judge so they run without any API key.
They prove the metric catches a claim attributed to the wrong document-native
label (e.g. "Table 3" when the fact actually appears under "Table 2"), which
FaithfulnessMetric would pass because the fact is true somewhere in context,
and CitationFaithfulnessMetric would not even parse (it only understands `[N]`
markers, not named labels).
"""

from deepeval.metrics.community import NamedReferenceAttributionMetric
from deepeval.metrics.community.named_reference_attribution.schema import (
    NamedReference,
    NamedReferences,
    NamedReferenceVerdict,
    Verdicts,
    NamedReferenceAttributionScoreReason,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase


class FakeJudge(DeepEvalBaseLLM):
    """Returns preset responses from a queue, one per call, keyed by schema."""

    def __init__(self, responses):
        # responses: list of objects to return, in call order
        self._responses = list(responses)
        self.prompts = []
        super().__init__(model="fake-judge")

    def load_model(self, *args, **kwargs):
        return None

    def generate(self, prompt, *args, schema=None, **kwargs):
        self.prompts.append(prompt)
        return self._responses.pop(0)

    async def a_generate(self, prompt, *args, schema=None, **kwargs):
        self.prompts.append(prompt)
        return self._responses.pop(0)

    def get_model_name(self, *args, **kwargs):
        return "fake-judge"


QUERY = "What does Table 3 say about EMEA revenue, and what does Section 4.2 say about notice?"
RETRIEVAL_CONTEXT = [
    "Table 2: Revenue by region, 2025. EMEA revenue was $12M.",
    "Table 3: Revenue by region, 2026. EMEA revenue was $15M.",
    "Section 4.2: Either party may terminate with 30 days written notice.",
]


def test_fails_wrong_table_attribution():
    judge = FakeJudge(
        [
            NamedReferences(
                references=[
                    NamedReference(
                        label="Table 3", claim="EMEA revenue was $12M"
                    )
                ]
            ),
            Verdicts(
                verdicts=[
                    NamedReferenceVerdict(
                        label="Table 3",
                        verdict="no",
                        reason="Table 3 states EMEA revenue was $15M; $12M appears under Table 2.",
                    )
                ]
            ),
            NamedReferenceAttributionScoreReason(
                reason="The EMEA revenue figure is attributed to Table 3 but actually appears under Table 2."
            ),
        ]
    )
    metric = NamedReferenceAttributionMetric(model=judge, async_mode=False)
    test_case = LLMTestCase(
        input=QUERY,
        actual_output="According to Table 3, EMEA revenue was $12M.",
        retrieval_context=RETRIEVAL_CONTEXT,
    )

    metric.measure(test_case)

    assert metric.score == 0.0
    assert metric.is_successful() is False
    assert metric.reason is not None


def test_passes_correct_attribution():
    judge = FakeJudge(
        [
            NamedReferences(
                references=[
                    NamedReference(
                        label="Table 3", claim="EMEA revenue was $15M"
                    ),
                    NamedReference(
                        label="Section 4.2",
                        claim="30 day notice period",
                    ),
                ]
            ),
            Verdicts(
                verdicts=[
                    NamedReferenceVerdict(label="Table 3", verdict="yes"),
                    NamedReferenceVerdict(label="Section 4.2", verdict="yes"),
                ]
            ),
            NamedReferenceAttributionScoreReason(
                reason="Every named reference matches its labeled content."
            ),
        ]
    )
    metric = NamedReferenceAttributionMetric(model=judge, async_mode=False)
    test_case = LLMTestCase(
        input=QUERY,
        actual_output=(
            "According to Table 3, EMEA revenue was $15M, and per Section "
            "4.2, notice is 30 days."
        ),
        retrieval_context=RETRIEVAL_CONTEXT,
    )

    metric.measure(test_case)

    assert metric.score == 1.0
    assert metric.is_successful() is True


def test_no_named_references_scores_one():
    judge = FakeJudge(
        [
            NamedReferences(references=[]),
            NamedReferenceAttributionScoreReason(
                reason="The answer made no named structural references."
            ),
        ]
    )
    metric = NamedReferenceAttributionMetric(model=judge, async_mode=False)
    test_case = LLMTestCase(
        input=QUERY,
        actual_output="EMEA revenue grew year over year.",
        retrieval_context=RETRIEVAL_CONTEXT,
    )

    metric.measure(test_case)

    assert metric.score == 1.0
    assert metric.is_successful() is True


def test_async_measure_matches_sync():
    judge = FakeJudge(
        [
            NamedReferences(
                references=[
                    NamedReference(
                        label="Table 3", claim="EMEA revenue was $12M"
                    )
                ]
            ),
            Verdicts(
                verdicts=[
                    NamedReferenceVerdict(
                        label="Table 3",
                        verdict="no",
                        reason="$12M appears under Table 2, not Table 3.",
                    )
                ]
            ),
            NamedReferenceAttributionScoreReason(reason="Misattributed."),
        ]
    )
    metric = NamedReferenceAttributionMetric(model=judge, async_mode=True)
    test_case = LLMTestCase(
        input=QUERY,
        actual_output="According to Table 3, EMEA revenue was $12M.",
        retrieval_context=RETRIEVAL_CONTEXT,
    )

    metric.measure(test_case)

    assert metric.score == 0.0
    assert metric.is_successful() is False
