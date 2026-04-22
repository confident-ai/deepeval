"""Demo: trace-scope assert_test inside `deepeval test run`."""

import pytest

from deepeval import assert_test
from deepeval.dataset import Golden
from deepeval.tracing import observe, update_current_trace
from deepeval.metrics import AnswerRelevancyMetric


@observe(metrics=[AnswerRelevancyMetric()])
def retriever(query: str) -> list[str]:
    return [f"chunk about: {query}", "static context chunk"]


@observe()
def llm_app(query: str) -> str:
    chunks = retriever(query)
    answer = f"stubbed answer for '{query}' using {len(chunks)} chunks"
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
    assert_test(golden=golden)
