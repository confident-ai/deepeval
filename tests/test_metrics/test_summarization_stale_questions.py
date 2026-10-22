"""A reused SummarizationMetric must regenerate auto-generated questions.

`assessment_questions` is auto-generated from the current test case's `input`
whenever the user did not supply their own. The generation twins only checked
`is None`, so after the first measure cached the questions, every later
measure on the same instance scored coverage against questions mined from the
FIRST input — a reused metric returned 0 where a fresh instance returned 1.
User-supplied questions are configuration and must survive across measures.
"""

import json

import pytest

from deepeval.metrics.summarization import SummarizationMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase


class TopicStubModel(DeepEvalBaseLLM):
    """Deterministic judge.

    Questions quote the topic marker found in the input text (PARIS/TOKYO);
    answers are "yes" only when the text mentions the topic the question asks
    about, so scoring against stale questions is observable.
    """

    def __init__(self):
        super().__init__(model="stub-sum")
        self.question_topics: list[str | None] = []

    def load_model(self):
        return None

    def get_model_name(self):
        return "stub-sum"

    @staticmethod
    def _topic(prompt: str):
        for marker in ("PARIS", "TOKYO"):
            if marker in prompt:
                return marker
        return None

    def generate(self, prompt: str, schema=None, **kwargs):
        if "closed-ended questions" in prompt:
            topic = self._topic(prompt)
            self.question_topics.append(topic)
            return json.dumps(
                {"questions": [f"Does the text discuss {topic}?"]}
            )
        if "key 'answers'" in prompt:
            before, after = prompt.rsplit("Questions:", 1)
            ok = self._topic(before) is not None and self._topic(
                before
            ) == self._topic(after)
            return json.dumps({"answers": ["yes" if ok else "no"]})
        if "summary claims" in prompt:
            return json.dumps({"verdicts": [{"verdict": "yes"}]})
        if "actual AI output" in prompt:
            return json.dumps({"claims": ["a claim"]})
        if "truths" in prompt[:400]:
            return json.dumps({"truths": ["a truth"]})
        return json.dumps({"reason": "fine"})

    async def a_generate(self, prompt: str, schema=None, **kwargs):
        return self.generate(prompt, schema=schema)


PARIS = LLMTestCase(
    input="All about PARIS and its cafes.", actual_output="PARIS summary."
)
TOKYO = LLMTestCase(
    input="All about TOKYO and its sushi.", actual_output="TOKYO summary."
)


def _metric(async_mode: bool) -> SummarizationMetric:
    return SummarizationMetric(
        model=TopicStubModel(),
        async_mode=async_mode,
        include_reason=False,
        verbose_mode=False,
    )


@pytest.mark.parametrize("async_mode", [False, True])
def test_reused_metric_regenerates_questions_per_input(async_mode):
    metric = _metric(async_mode)

    first = metric.measure(PARIS)
    assert first == 1.0, "PARIS case scored against PARIS questions"

    second = metric.measure(TOKYO)
    assert metric.assessment_questions == ["Does the text discuss TOKYO?"], (
        "questions must be regenerated from the current input, not reused from "
        f"the first measure: {metric.assessment_questions!r}"
    )
    assert metric.model.question_topics == [
        "PARIS",
        "TOKYO",
    ], "the question generator must run once per measure"
    assert second == 1.0, (
        f"a reused metric must score the TOKYO case like a fresh instance "
        f"(1.0), got {second}"
    )


def test_user_supplied_questions_are_kept_across_measures():
    metric = SummarizationMetric(
        model=TopicStubModel(),
        async_mode=False,
        include_reason=False,
        verbose_mode=False,
        assessment_questions=["Is the summary faithful?"],
    )

    metric.measure(PARIS)
    metric.measure(TOKYO)

    assert metric.assessment_questions == ["Is the summary faithful?"]
    assert (
        metric.model.question_topics == []
    ), "user questions must never be regenerated"
