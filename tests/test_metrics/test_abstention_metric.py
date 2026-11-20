"""Tests for AbstentionMetric.

These tests use a fake DeepEvalBaseLLM judge so they run without any API key.
They cover all four decision quadrants — correct answer, correct abstention,
over-refusal, and unsupported answer — plus async/sync parity.
"""

from deepeval.metrics.community import AbstentionMetric
from deepeval.metrics.community.abstention.schema import AbstentionVerdict
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase


class FakeJudge(DeepEvalBaseLLM):
    """Returns a preset verdict, capturing the prompt it was given."""

    def __init__(self, verdict: AbstentionVerdict):
        self._verdict = verdict
        self.last_prompt = None
        super().__init__(model="fake-judge")

    def load_model(self, *args, **kwargs):
        return None

    def generate(self, prompt, *args, schema=None, **kwargs):
        self.last_prompt = prompt
        return self._verdict

    async def a_generate(self, prompt, *args, schema=None, **kwargs):
        self.last_prompt = prompt
        return self._verdict

    def get_model_name(self, *args, **kwargs):
        return "fake-judge"


QUERY = "What is the Eiffel Tower's daily visitor capacity?"
# Neither passage covers daily visitor capacity — the question is unanswerable
# from this context.
UNSUPPORTED_CONTEXT = [
    "The Eiffel Tower stands 330 metres tall in Paris.",
    "The Eiffel Tower was completed in 1889 for the World Fair.",
]


def test_passes_appropriate_abstention():
    # Context does not support an answer, and the system abstained -> correct.
    judge = FakeJudge(
        AbstentionVerdict(
            context_supports_answer=False,
            output_abstained=True,
            reasoning="The passages never mention visitor capacity and the answer declines.",
        )
    )
    metric = AbstentionMetric(model=judge, async_mode=False)
    test_case = LLMTestCase(
        input=QUERY,
        actual_output="The context doesn't state the Eiffel Tower's daily visitor capacity.",
        retrieval_context=UNSUPPORTED_CONTEXT,
    )

    metric.measure(test_case)

    assert metric.score == 1.0
    assert metric.is_successful() is True
    assert metric.reason is not None
    # The prompt must number the passages.
    assert "[1] The Eiffel Tower stands 330 metres tall" in judge.last_prompt
    assert "[2] The Eiffel Tower was completed in 1889" in judge.last_prompt


def test_fails_unsupported_answer():
    # Context does not support an answer, but the system answered anyway -> wrong.
    judge = FakeJudge(
        AbstentionVerdict(
            context_supports_answer=False,
            output_abstained=False,
            reasoning="The passages never mention visitor capacity but the answer states a number.",
        )
    )
    metric = AbstentionMetric(model=judge, async_mode=False)
    test_case = LLMTestCase(
        input=QUERY,
        actual_output="The Eiffel Tower welcomes about 25,000 visitors per day.",
        retrieval_context=UNSUPPORTED_CONTEXT,
    )

    metric.measure(test_case)

    assert metric.score == 0.0
    assert metric.is_successful() is False


def test_fails_over_refusal():
    # Context supports an answer, but the system abstained anyway -> over-refusal.
    judge = FakeJudge(
        AbstentionVerdict(
            context_supports_answer=True,
            output_abstained=True,
            reasoning="The height is stated in the passages but the answer declines to give it.",
        )
    )
    metric = AbstentionMetric(model=judge, async_mode=False)
    test_case = LLMTestCase(
        input="How tall is the Eiffel Tower?",
        actual_output="I'm not able to find that in the provided context.",
        retrieval_context=UNSUPPORTED_CONTEXT,
    )

    metric.measure(test_case)

    assert metric.score == 0.0
    assert metric.is_successful() is False


def test_passes_answered_when_supported():
    # Context supports an answer, and the system answered -> correct.
    judge = FakeJudge(
        AbstentionVerdict(
            context_supports_answer=True,
            output_abstained=False,
            reasoning="The height is stated in the passages and the answer gives it.",
        )
    )
    metric = AbstentionMetric(model=judge, async_mode=False)
    test_case = LLMTestCase(
        input="How tall is the Eiffel Tower?",
        actual_output="The Eiffel Tower stands 330 metres tall.",
        retrieval_context=UNSUPPORTED_CONTEXT,
    )

    metric.measure(test_case)

    assert metric.score == 1.0
    assert metric.is_successful() is True


def test_async_measure_matches_sync():
    judge = FakeJudge(
        AbstentionVerdict(
            context_supports_answer=False,
            output_abstained=True,
            reasoning="unanswerable, correctly declined",
        )
    )
    metric = AbstentionMetric(model=judge, async_mode=True)
    test_case = LLMTestCase(
        input=QUERY,
        actual_output="The context doesn't state the Eiffel Tower's daily visitor capacity.",
        retrieval_context=UNSUPPORTED_CONTEXT,
    )

    metric.measure(test_case)

    assert metric.score == 1.0
    assert metric.is_successful() is True
