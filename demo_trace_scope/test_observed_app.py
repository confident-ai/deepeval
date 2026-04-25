"""Demo: trace-scope assert_test inside `deepeval test run`."""

import pytest

from deepeval import assert_test
from deepeval.dataset import Golden
from deepeval.tracing import observe, update_current_span, update_current_trace
from deepeval.metrics import AnswerRelevancyMetric, GEval, FaithfulnessMetric
from deepeval.test_case import LLMTestCaseParams


@observe(metrics=[AnswerRelevancyMetric()])
def retriever(query: str) -> list[str]:
    return [f"chunk about: {query}", "static context chunk"]

metric1 = GEval(
    name="Metric 1",
    criteria="Metric 1 criteria",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
)

@observe(metrics=[FaithfulnessMetric()])
def llm_app(query: str) -> str:
    chunks = retriever(query)
    answer = f"stubbed answer for '{query}' using {len(chunks)} chunks"
    update_current_span(retrieval_context=chunks)
    update_current_trace(
        input=query,
        output=answer,
        retrieval_context=chunks,
    )
    return answer


GOLDENS = [
    Golden(input="What is the capital of France?"),
    Golden(input="Who wrote Hamlet?"),
]


@pytest.mark.parametrize("golden", GOLDENS)
def test_llm_app_trace_scope(golden: Golden):
    llm_app(golden.input)
    assert_test(golden=golden, metrics=[metric1])
