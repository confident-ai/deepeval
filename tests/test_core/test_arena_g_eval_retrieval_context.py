import json

from deepeval.metrics.arena_g_eval.utils import format_arena_test_case
from deepeval.test_case import (
    ArenaTestCase,
    Contestant,
    LLMTestCase,
    RetrievedContextData,
    SingleTurnParams,
)


def test_arena_prompt_formats_structured_retrieval_context():
    contestants = [
        Contestant(
            name=name,
            test_case=LLMTestCase(
                input="What is the refund window?",
                actual_output="30 days.",
                retrieval_context=[
                    RetrievedContextData(
                        context="Refunds are accepted for 30 days.",
                        source="policy.md",
                    ),
                    "Plain string chunk.",
                ],
            ),
        )
        for name in ("a", "b")
    ]

    formatted, _ = format_arena_test_case(
        [SingleTurnParams.RETRIEVAL_CONTEXT],
        ArenaTestCase(contestants=contestants),
    )

    prompt = json.loads(repr(formatted))
    for contestant in prompt["arena_test_cases"].values():
        assert json.loads(contestant)["retrieval_context"] == [
            "policy.md: Refunds are accepted for 30 days.",
            "Plain string chunk.",
        ]
