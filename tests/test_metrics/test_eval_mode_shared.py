"""Shared System One plumbing that is not tied to one metric: which metrics
can honour `system_one`, the DAG running as `hybrid`, the trace sent to Jev
and the extra state a spec adds. No network: Jev and the LLM are fakes."""

import inspect

import pytest

import deepeval.metrics as metrics_module
from deepeval.errors import DeepEvalError
from deepeval.metrics import BaseConversationalMetric, BaseMetric, DAGMetric
from deepeval.metrics.dag import (
    BinaryJudgementNode,
    DeepAcyclicGraph,
    VerdictNode,
)
from deepeval.metrics.utils import (
    compact_trace,
    effective_eval_mode,
    has_whole_metric_form,
)
from deepeval.config.eval_mode import EvalMode
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase

from tests.test_metrics.system_one_fakes import (
    ExplodingSystemOneModel,
    FakeSystemOneModel,
    answer_everything,
    noul_answers,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in ("DEEPEVAL_MODE", "DEEPEVAL_EVAL_MODE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DEEPEVAL_DISABLE_LEGACY_KEYFILE", "1")
    yield


###############################################
# Which metrics honour `system_one`
###############################################

# Metrics that take a System One model but have no whole-metric form: the
# DAGs (task nodes need the LLM) and JevEval (Jev by design).
NO_WHOLE_METRIC_FORM = {
    "DAGMetric",
    "ConversationalDAGMetric",
    "JevEval",
    "ConversationalJevEval",
}


def _metric_classes():
    for name in dir(metrics_module):
        obj = getattr(metrics_module, name)
        if (
            inspect.isclass(obj)
            and issubclass(obj, (BaseMetric, BaseConversationalMetric))
            and obj not in (BaseMetric, BaseConversationalMetric)
        ):
            yield obj


def _takes(cls, name):
    return name in inspect.signature(cls.__init__).parameters


def test_every_system_one_metric_uses_its_system_one_model():
    missing = [
        cls.__name__
        for cls in _metric_classes()
        if _takes(cls, "system_one_model")
        and cls.__name__ not in NO_WHOLE_METRIC_FORM
        and not has_whole_metric_form(object.__new__(cls))
    ]
    assert missing == []


def test_every_system_one_metric_takes_eval_mode():
    missing = [
        cls.__name__
        for cls in _metric_classes()
        if _takes(cls, "system_one_model") and not _takes(cls, "eval_mode")
        and cls.__name__ not in {"JevEval", "ConversationalJevEval"}
    ]
    assert missing == []


def test_json_correctness_has_no_system_one_model():
    from deepeval.metrics import JsonCorrectnessMetric

    assert not _takes(JsonCorrectnessMetric, "system_one_model")
    assert _takes(JsonCorrectnessMetric, "eval_mode")


###############################################
# DAG under `system_one` runs as `hybrid`
###############################################


class DagLLM(DeepEvalBaseLLM):
    def __init__(self):
        self.calls = 0
        super().__init__(model="dag-llm")

    def load_model(self):
        return self

    def generate(self, prompt, schema=None, **kwargs):
        self.calls += 1
        return schema(verdict=True, reason="llm says yes")

    async def a_generate(self, prompt, schema=None, **kwargs):
        return self.generate(prompt, schema=schema)

    def get_model_name(self):
        return "dag-llm"


def _dag(**kwargs):
    node = BinaryJudgementNode(
        criteria="Does `actual_output` answer `input`?",
        children=[
            VerdictNode(verdict=False, score=0),
            VerdictNode(verdict=True, score=10),
        ],
    )
    return DAGMetric(
        name="Answers",
        dag=DeepAcyclicGraph(root_nodes=[node]),
        async_mode=False,
        include_reason=False,
        **kwargs,
    )


TEST_CASE = LLMTestCase(input="What is 2 + 2?", actual_output="4")


def test_dag_under_system_one_runs_as_hybrid():
    metric = _dag(
        model=DagLLM(),
        system_one_model=FakeSystemOneModel(answer_fn=noul_answers(0.9)),
        eval_mode="system_one",
    )
    assert effective_eval_mode(metric) is EvalMode.HYBRID
    assert metric.model is not None
    assert metric.measure(TEST_CASE) == 1.0
    assert len(metric.system_one_model.calls) == 1
    assert metric.model.calls == 0


def test_dag_under_system_one_falls_back_to_llm_on_jev_error():
    llm = DagLLM()
    metric = _dag(
        model=llm,
        system_one_model=ExplodingSystemOneModel(ConnectionError("down")),
        eval_mode="system_one",
    )
    assert metric.measure(TEST_CASE) == 1.0
    assert llm.calls >= 1
    assert "down" in metric.system_one_fallback_reason


def test_wired_metric_keeps_system_one():
    from deepeval.metrics import AnswerRelevancyMetric

    metric = AnswerRelevancyMetric(
        system_one_model=FakeSystemOneModel(answer_fn=answer_everything()),
        eval_mode="system_one",
    )
    assert effective_eval_mode(metric) is EvalMode.SYSTEM_ONE


###############################################
# Trace sent to Jev
###############################################

TRACE = {
    "name": "trip_planner",
    "type": "agent",
    "input": {"input": "Plan a trip"},
    "output": "Booked",
    "inputTokenCount": 120,
    "integration": "langgraph",
    "available_tools": [],
    "children": [
        {
            "name": "flight_tool",
            "type": "tool",
            "input": {"to": "NYC"},
            "output": ["F1"],
            "model": None,
            "costPerInputToken": 0.1,
            "children": [],
        }
    ],
}


def test_compact_trace_keeps_what_the_agent_did():
    assert compact_trace(TRACE) == {
        "name": "trip_planner",
        "type": "agent",
        "input": {"input": "Plan a trip"},
        "output": "Booked",
        "children": [
            {
                "name": "flight_tool",
                "type": "tool",
                "input": {"to": "NYC"},
                "output": ["F1"],
            }
        ],
    }


def _task_completion(**kwargs):
    from deepeval.metrics import TaskCompletionMetric

    kwargs.setdefault("async_mode", False)
    return TaskCompletionMetric(**kwargs)


def _traced(trace):
    tc = LLMTestCase(input="Plan a trip", actual_output="Booked")
    tc._trace_dict = trace
    return tc


def test_trace_goes_to_jev_as_extra_state():
    jev = FakeSystemOneModel(answer_fn=answer_everything())
    metric = _task_completion(system_one_model=jev, eval_mode="system_one")
    metric.measure(_traced(TRACE))
    state, _ = jev.calls[0]
    assert state == {"trace": compact_trace(TRACE)}


def test_user_task_goes_to_jev_and_questions():
    jev = FakeSystemOneModel(answer_fn=answer_everything())
    metric = _task_completion(
        task="Plan a trip to NYC", system_one_model=jev, eval_mode="system_one"
    )
    metric.measure(_traced(TRACE))
    state, questions = jev.calls[0]
    assert state["task"] == "Plan a trip to NYC"
    assert all("`task`" in str(q.instructions) for q in questions.values())


def test_oversized_trace_says_switch_to_llm():
    huge = {**TRACE, "output": "x " * 200_000}
    jev = FakeSystemOneModel(answer_fn=answer_everything())
    metric = _task_completion(system_one_model=jev, eval_mode="system_one")
    with pytest.raises(DeepEvalError, match='eval_mode="llm"'):
        metric.measure(_traced(huge))
    assert jev.calls == []
