"""Offline coverage from judge verdicts through public metric outcomes."""

import json

import pytest

from deepeval.metrics import BiasMetric, PromptAlignmentMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase


class OfflineVerdictJudge(DeepEvalBaseLLM):
    def __init__(self, structured):
        self.structured = structured
        self.calls = []
        super().__init__(model="offline-verdict-judge")

    def load_model(self):
        return None

    def get_model_name(self):
        return "offline-verdict-judge"

    def _respond(self, schema, asynchronous):
        self.calls.append((tuple(schema.model_fields), asynchronous))
        if "opinions" in schema.model_fields:
            payload = {"opinions": ["first", "second", "third"]}
        elif "verdicts" in schema.model_fields:
            values = (
                ["yes", "yes", "no"]
                if self.structured
                else [
                    " Yes, first finding.",
                    "YES - second finding.",
                    "No, third finding.",
                ]
            )
            payload = {
                "verdicts": [
                    {"verdict": value, "reason": f"finding {index}"}
                    for index, value in enumerate(values)
                ]
            }
        else:
            assert "reason" in schema.model_fields
            payload = {"reason": "Offline judge summary."}
        return schema(**payload) if self.structured else json.dumps(payload)

    def generate(self, prompt, schema=None):
        return self._respond(schema, asynchronous=False)

    async def a_generate(self, prompt, schema=None):
        return self._respond(schema, asynchronous=True)


@pytest.mark.parametrize("metric_kind", ["bias", "alignment"])
@pytest.mark.parametrize("path", ["sync", "measure_async", "a_measure"])
@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("strict", [False, True])
def test_public_measurement_preserves_verdict_polarity(
    metric_kind, path, structured, strict
):
    import asyncio

    judge = OfflineVerdictJudge(structured)
    kwargs = dict(
        model=judge,
        eval_mode="llm",
        async_mode=path != "sync",
        strict_mode=strict,
    )
    metric = (
        BiasMetric(**kwargs)
        if metric_kind == "bias"
        else PromptAlignmentMetric(
            prompt_instructions=["first", "second", "third"], **kwargs
        )
    )
    case = LLMTestCase(input="Question", actual_output="Answer")

    score = (
        asyncio.run(metric.a_measure(case, _show_indicator=False))
        if path == "a_measure"
        else metric.measure(case, _show_indicator=False)
    )

    # The same two yes + one no verdicts have opposite scoring polarity:
    # yes identifies bias, but yes confirms prompt alignment.
    expected = 0 if strict else (1 / 3 if metric_kind == "bias" else 2 / 3)
    assert score == pytest.approx(expected)
    assert metric.score == pytest.approx(expected)
    assert metric.success is (not strict and metric_kind == "alignment")
    assert metric.is_successful() is metric.success
    assert [str(v.verdict) for v in metric.verdicts] == ["yes", "yes", "no"]
    assert [v.reason for v in metric.verdicts] == [
        "finding 0",
        "finding 1",
        "finding 2",
    ]
    assert metric.reason == "Offline judge summary."
    assert len(judge.calls) == (3 if metric_kind == "bias" else 2)
    assert all(
        asynchronous == (path != "sync") for _, asynchronous in judge.calls
    )
