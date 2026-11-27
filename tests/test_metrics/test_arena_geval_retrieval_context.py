import json

import pytest

from deepeval.metrics import ArenaGEval
from deepeval.metrics.arena_g_eval.schema import RewrittenReason, Winner
from deepeval.metrics.arena_g_eval.utils import (
    FAKE_NAMES,
    format_arena_test_case,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import (
    ArenaTestCase,
    Contestant,
    LLMTestCase,
    SingleTurnParams,
)
from deepeval.test_case.llm_test_case import RetrievedContextData

PARAMS = [
    SingleTurnParams.INPUT,
    SingleTurnParams.ACTUAL_OUTPUT,
    SingleTurnParams.RETRIEVAL_CONTEXT,
]
RETRIEVAL_CONTEXT = [
    RetrievedContextData(
        context="Paris is the capital of France.", source="wiki"
    ),
    "France is in Europe.",
]


class WinnerPromptModel(DeepEvalBaseLLM):
    """Deterministic judge that picks the first contestant it was shown
    and keeps the winner prompt so the test can read it."""

    def __init__(self):
        self.winner_prompt = None
        super().__init__(model="winner-prompt-model")

    def load_model(self):
        return self

    def get_model_name(self):
        return "winner-prompt-model"

    def generate(self, prompt, schema=None):
        if schema is RewrittenReason:
            return RewrittenReason(rewritten_reason="The first is better.")
        if schema is not Winner:
            raise AssertionError(f"unexpected schema {schema}")
        self.winner_prompt = prompt
        # Read the names from the test case, not the template's examples.
        contents = prompt[prompt.index('"arena_test_cases"') :]
        first = min(
            (contents.index(f'"{name}"'), name)
            for name in FAKE_NAMES
            if f'"{name}"' in contents
        )[1]
        return Winner(winner=first, reason=f"{first} is better.")

    async def a_generate(self, prompt, schema=None):
        return self.generate(prompt, schema)


def _arena_test_case():
    return ArenaTestCase(
        contestants=[
            Contestant(
                name=name,
                test_case=LLMTestCase(
                    input="What is the capital of France?",
                    actual_output=output,
                    retrieval_context=RETRIEVAL_CONTEXT,
                ),
            )
            for name, output in (("v1", "Paris."), ("v2", "It is Paris."))
        ]
    )


def test_retrieved_context_data_is_rendered_as_source_and_context():
    formatted, _ = format_arena_test_case(PARAMS, _arena_test_case())

    data = json.loads(repr(formatted))

    for contestant in data["arena_test_cases"].values():
        # Contestants are nested as JSON strings until #3789 merges.
        if isinstance(contestant, str):
            contestant = json.loads(contestant)
        assert contestant["retrieval_context"] == [
            "wiki: Paris is the capital of France.",
            "France is in Europe.",
        ]


@pytest.mark.parametrize("async_mode", [False, True])
def test_measure_accepts_retrieved_context_data(async_mode):
    judge = WinnerPromptModel()
    metric = ArenaGEval(
        name="Faithfulness",
        evaluation_steps=["Pick the output more faithful to the context."],
        evaluation_params=PARAMS,
        model=judge,
        async_mode=async_mode,
    )

    metric.measure(_arena_test_case())

    assert metric.winner in ("v1", "v2")
    assert "wiki: Paris is the capital of France." in judge.winner_prompt
