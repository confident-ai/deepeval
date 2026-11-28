import json

from deepeval.metrics.arena_g_eval.utils import format_arena_test_case
from deepeval.test_case import (
    ArenaTestCase,
    Contestant,
    LLMTestCase,
    SingleTurnParams,
)
from deepeval.test_case.llm_test_case import RetrievedContextData


def _arena_test_case(retrieval_context):
    return ArenaTestCase(
        contestants=[
            Contestant(
                name="v1",
                test_case=LLMTestCase(
                    input="What is the capital of France?",
                    actual_output="Paris.",
                    retrieval_context=retrieval_context,
                ),
            ),
            Contestant(
                name="v2",
                test_case=LLMTestCase(
                    input="What is the capital of France?",
                    actual_output="It is Paris.",
                    retrieval_context=retrieval_context,
                ),
            ),
        ]
    )


def _formatted_retrieval_contexts(retrieval_context):
    formatted, _ = format_arena_test_case(
        [
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.RETRIEVAL_CONTEXT,
        ],
        _arena_test_case(retrieval_context),
    )
    parsed = json.loads(repr(formatted))
    contexts = []
    for contestant in parsed["arena_test_cases"].values():
        inner = json.loads(contestant)
        contexts.append(inner["retrieval_context"])
    return contexts


def test_retrieval_context_with_retrieved_context_data():
    contexts = _formatted_retrieval_contexts(
        [
            RetrievedContextData(
                context="Paris is the capital of France.", source="wiki"
            )
        ]
    )
    assert contexts == [
        ["wiki: Paris is the capital of France."],
        ["wiki: Paris is the capital of France."],
    ]


def test_retrieval_context_with_plain_strings_unchanged():
    contexts = _formatted_retrieval_contexts(
        ["Paris is the capital of France.", "France is in Europe."]
    )
    assert contexts == [
        ["Paris is the capital of France.", "France is in Europe."],
        ["Paris is the capital of France.", "France is in Europe."],
    ]
