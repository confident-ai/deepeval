"""Regression tests: metrics must accept ``RetrievedContextData`` items in
``retrieval_context``.

``LLMTestCase.retrieval_context`` and ``Turn.retrieval_context`` are typed
``List[Union[str, RetrievedContextData]]``, but FaithfulnessMetric,
ContextualRelevancyMetric, TurnFaithfulnessMetric and
TurnContextualRelevancyMetric joined or split the items as plain strings and
crashed with ``TypeError`` before calling the judge. These run offline with a canned judge.
"""

import json

import pytest

from deepeval.metrics import (
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    TurnContextualRelevancyMetric,
    TurnFaithfulnessMetric,
)
from deepeval.metrics.retrieval_context_display import (
    retrieval_context_to_strings,
)
from deepeval.test_case import ConversationalTestCase, LLMTestCase, Turn
from deepeval.test_case.llm_test_case import RetrievedContextData
from tests.test_metrics.system_one_fakes import CannedLLM

FACT = "Paris is the capital of France."
RENDERED = f"wiki: {FACT}"

# One reply that satisfies every schema these metrics parse along the way.
CANNED_REPLY = json.dumps(
    {
        "truths": [FACT],
        "claims": ["Paris is the capital of France."],
        "verdicts": [{"verdict": "yes", "reason": "ok", "statement": FACT}],
        "reason": "ok",
    }
)


def _structured_context():
    return [RetrievedContextData(context=FACT, source="wiki")]


def test_retrieval_context_to_strings_uses_the_type_serializer():
    mixed = ["plain chunk", RetrievedContextData(context=FACT, source="wiki")]

    assert retrieval_context_to_strings(mixed) == ["plain chunk", RENDERED]


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize(
    "metric_cls", [FaithfulnessMetric, ContextualRelevancyMetric]
)
def test_single_turn_metrics_accept_retrieved_context_data(
    metric_cls, async_mode
):
    llm = CannedLLM(CANNED_REPLY)
    metric = metric_cls(model=llm, eval_mode="llm", async_mode=async_mode)

    metric.measure(
        LLMTestCase(
            input="What is the capital of France?",
            actual_output="Paris is the capital of France.",
            retrieval_context=_structured_context(),
        )
    )

    assert metric.score == 1.0
    assert any(RENDERED in str(p) for p in llm.prompts)


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize(
    "metric_class", [TurnFaithfulnessMetric, TurnContextualRelevancyMetric]
)
def test_turn_metrics_accept_retrieved_context_data(metric_class, async_mode):
    llm = CannedLLM(CANNED_REPLY)
    metric = metric_class(model=llm, eval_mode="llm", async_mode=async_mode)

    metric.measure(
        ConversationalTestCase(
            turns=[
                Turn(role="user", content="What is the capital of France?"),
                Turn(
                    role="assistant",
                    content="Paris is the capital of France.",
                    retrieval_context=_structured_context(),
                ),
            ]
        )
    )

    assert metric.score == 1.0
    assert any(RENDERED in str(p) for p in llm.prompts)
