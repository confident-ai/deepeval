from __future__ import annotations

import os

import pytest

from deepeval.metrics import ArenaGEval, ConversationalGEval, DAGMetric, GEval
from deepeval.metrics.dag import BinaryJudgementNode, DeepAcyclicGraph, VerdictNode
from deepeval.test_case import (
    ArenaTestCase,
    Contestant,
    ConversationalTestCase,
    LLMTestCase,
    SingleTurnParams,
    Turn,
)

pytestmark = pytest.mark.skipif(
    os.getenv("OPENAI_API_KEY") is None
    or not os.getenv("OPENAI_API_KEY").strip(),
    reason="OPENAI_API_KEY is not set",
)


def test_geval_measure_smoke():
    test_case = LLMTestCase(
        input="What is the capital of France?",
        actual_output="Paris is the capital.",
    )
    metric = GEval(
        name="SurveyGEval",
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
        ],
        criteria="Check that the answer is factually relevant to the question.",
        async_mode=False,
    )
    metric.measure(test_case, _show_indicator=False, _log_metric_to_confident=False)
    assert metric.score is not None
    assert metric.reason is not None


def test_conversational_geval_measure_smoke():
    convo = ConversationalTestCase(
        turns=[
            Turn(role="user", content="Hello, I need help with a refund."),
            Turn(
                role="assistant",
                content="I can help you start a refund request.",
            ),
        ],
        expected_outcome="The assistant should help with refunds politely.",
        chatbot_role="Support agent",
    )
    metric = ConversationalGEval(
        name="SurveyConversationalGEval",
        evaluation_params=None,
        criteria="Check that the assistant stays on topic and is helpful.",
        async_mode=False,
    )
    metric.measure(convo, _show_indicator=False, _log_metric_to_confident=False)
    assert metric.score is not None
    assert metric.reason is not None


def test_arena_geval_measure_smoke():
    shared_input = "What is 2 + 2?"
    arena = ArenaTestCase(
        contestants=[
            Contestant(
                name="ModelA",
                test_case=LLMTestCase(
                    input=shared_input,
                    actual_output="test: answer A says four",
                    expected_output=None,
                ),
            ),
            Contestant(
                name="ModelB",
                test_case=LLMTestCase(
                    input=shared_input,
                    actual_output="test: answer B says 4",
                    expected_output=None,
                ),
            ),
        ]
    )
    metric = ArenaGEval(
        name="SurveyArenaGEval",
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
        ],
        criteria="Pick the contestant whose answer is clearer and correct.",
        async_mode=False,
    )
    winner = metric.measure(arena, _show_indicator=False)
    assert winner in ("ModelA", "ModelB")
    assert metric.reason


def test_dag_metric_measure_smoke():
    metric = DAGMetric(
        name="SurveyDAG",
        dag=DeepAcyclicGraph(
            root_nodes=[
                BinaryJudgementNode(
                    criteria="Does the actual output mention a digit or number word?",
                    evaluation_params=[
                        SingleTurnParams.INPUT,
                        SingleTurnParams.ACTUAL_OUTPUT,
                    ],
                    children=[
                        VerdictNode(verdict=False, score=0),
                        VerdictNode(verdict=True, score=10),
                    ],
                )
            ]
        ),
        async_mode=False,
        include_reason=False,
    )
    metric.measure(
        LLMTestCase(
            input="Count",
            actual_output="The number 3 appears once.",
        ),
        _show_indicator=False,
        _log_metric_to_confident=False,
    )
    assert metric.score is not None
