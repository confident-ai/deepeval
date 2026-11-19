"""GEval without `evaluation_params` judges the trace (`_trace_dict`)
instead of individual test case fields."""

import pytest

from deepeval.errors import MissingTestCaseParamsError
from deepeval.metrics import GEval
from deepeval.metrics.g_eval.schema import ReasonScore, Steps
from deepeval.metrics.g_eval.utils import MetricPullResponse
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, SingleTurnParams


TRACE = {
    "name": "support_agent",
    "type": "agent",
    "input": {"input": "Where is my order #1234?"},
    "output": "Your order is in transit.",
    "children": [
        {
            "name": "order_lookup",
            "type": "tool",
            "input": {"order_id": "1234"},
            "output": {"status": "in_transit"},
            "children": [],
        }
    ],
}


class RecordingJudge(DeepEvalBaseLLM):
    def __init__(self):
        self.prompts = []
        super().__init__(model="recording-judge")

    def load_model(self, *args, **kwargs):
        return None

    def generate(self, prompt, schema=None, **kwargs):
        self.prompts.append(prompt)
        if schema is Steps:
            return Steps(steps=["Check the tool call.", "Check the output."])
        return ReasonScore(reason="order_lookup was used correctly.", score=8)

    async def a_generate(self, prompt, schema=None, **kwargs):
        return self.generate(prompt, schema=schema, **kwargs)

    def get_model_name(self, *args, **kwargs):
        return "recording-judge"


def _traced_test_case() -> LLMTestCase:
    test_case = LLMTestCase(input="None")
    test_case._trace_dict = TRACE
    return test_case


def test_requires_trace_follows_evaluation_params():
    assert GEval(name="t", criteria="c", model=RecordingJudge()).requires_trace
    assert not GEval(
        name="t",
        criteria="c",
        evaluation_params=[SingleTurnParams.INPUT],
        model=RecordingJudge(),
    ).requires_trace


@pytest.mark.parametrize("async_mode", [True, False])
@pytest.mark.parametrize("strict_mode", [False, True])
def test_measure_uses_trace_prompts(async_mode, strict_mode):
    judge = RecordingJudge()
    metric = GEval(
        name="Trajectory",
        criteria="Did the agent use the right tool and report its result?",
        model=judge,
        async_mode=async_mode,
        strict_mode=strict_mode,
    )

    metric.measure(_traced_test_case())

    steps_prompt, results_prompt = judge.prompts
    assert "execution trace" in steps_prompt
    assert '"name": "order_lookup"' in results_prompt
    assert "Test Case:" not in results_prompt
    assert metric.score is not None
    assert metric.reason == "order_lookup was used correctly."


def test_measure_without_trace_raises():
    metric = GEval(
        name="Trajectory",
        criteria="c",
        model=RecordingJudge(),
        async_mode=False,
    )
    with pytest.raises(MissingTestCaseParamsError, match="evaluates the trace"):
        metric.measure(LLMTestCase(input="hi", actual_output="hello"))


def test_upload_rejects_trajectory_mode():
    metric = GEval(name="t", criteria="c", model=RecordingJudge())
    with pytest.raises(ValueError, match="cannot be uploaded"):
        metric.upload()


def test_pull_with_params_leaves_trajectory_mode(monkeypatch):
    metric = GEval(name="t", model=RecordingJudge())
    assert metric.requires_trace

    response = MetricPullResponse(
        id="m1", criteria="c", requiredParameters=["input", "actualOutput"]
    )

    class FakeApi:
        def send_request(self, **kwargs):
            return response.model_dump(), None

    monkeypatch.setattr("deepeval.metrics.g_eval.g_eval.Api", FakeApi)
    metric.pull()

    assert metric.evaluation_params == [
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
    ]
    assert not metric.requires_trace
