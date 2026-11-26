"""Regression tests for how ArenaGEval resolves the judge's winner.

The judge is asked for "the exact contestant name", but nothing enforced it:
the answer was used as a raw dict key, so any drift ("iris", "Iris.",
"$Iris$", "Tie") killed `measure()` with a bare `KeyError`.

These tests run with a stub model, so they need no LLM provider API key.
"""

import json

import pytest

from deepeval.errors import DeepEvalError
from deepeval.metrics import ArenaGEval
from deepeval.metrics.arena_g_eval.utils import resolve_masked_winner
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import (
    ArenaTestCase,
    Contestant,
    LLMTestCase,
    SingleTurnParams,
)

NAMES = {"Alice": "model-alpha", "Bob": "model-beta"}

WINNING_OUTPUT = "Hello!"
WINNING_CONTESTANT = "model-alpha"


@pytest.mark.parametrize(
    "answer",
    ["Alice", "alice", "ALICE", " Alice\n", "Alice.", "$Alice$", '"Alice"'],
)
def test_resolves_harmless_drift_to_the_real_name(answer):
    assert resolve_masked_winner(answer, NAMES) == "model-alpha"


@pytest.mark.parametrize(
    "answer",
    ["Tie", "Alice and Bob", "Alice is the winner", "model-alpha", "", None, 1],
)
def test_unresolvable_winner_raises_deepeval_error(answer):
    with pytest.raises(DeepEvalError) as exc_info:
        resolve_masked_winner(answer, NAMES)

    assert repr(answer) in str(exc_info.value)
    assert "Alice" in str(exc_info.value)
    assert "Bob" in str(exc_info.value)


def _masked_name_of(prompt: str, actual_output: str) -> str:
    """Find the masked name the prompt gives to the contestant that produced
    `actual_output`. The masking is re-randomized on every run."""
    decoder = json.JSONDecoder()
    for start, char in enumerate(prompt):
        if char != "{":
            continue
        try:
            data, _ = decoder.raw_decode(prompt[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "arena_test_cases" in data:
            for masked_name, contestant in data["arena_test_cases"].items():
                if json.loads(contestant)["actual_output"] == actual_output:
                    return masked_name
    raise AssertionError("contestants not found in the judge prompt")


class _WinnerJudge(DeepEvalBaseLLM):
    """Picks the contestant that said WINNING_OUTPUT, then applies `mutate`
    to the masked name before answering."""

    def __init__(self, mutate, as_schema=False):
        self.mutate = mutate
        self.as_schema = as_schema

    def load_model(self, *args, **kwargs):
        return None

    def generate(self, prompt, schema=None, **kwargs):
        answers = {
            "Steps": {"steps": ["Pick the friendlier reply."]},
            "RewrittenReason": {"rewritten_reason": "stub"},
        }
        if schema.__name__ == "Winner":
            masked_name = _masked_name_of(prompt, WINNING_OUTPUT)
            answer = {"winner": self.mutate(masked_name), "reason": "stub"}
        else:
            answer = answers[schema.__name__]
        return schema(**answer) if self.as_schema else json.dumps(answer)

    async def a_generate(self, prompt, schema=None, **kwargs):
        return self.generate(prompt, schema=schema, **kwargs)

    def get_model_name(self, *args, **kwargs):
        return "winner-judge"


def _measure(mutate, async_mode, as_schema=False):
    metric = ArenaGEval(
        name="Friendly",
        criteria="Which reply is friendlier?",
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
        ],
        model=_WinnerJudge(mutate, as_schema=as_schema),
        async_mode=async_mode,
    )
    test_case = ArenaTestCase(
        contestants=[
            Contestant(
                name=WINNING_CONTESTANT,
                test_case=LLMTestCase(input="hi", actual_output=WINNING_OUTPUT),
            ),
            Contestant(
                name="model-beta",
                test_case=LLMTestCase(input="hi", actual_output="Hi there."),
            ),
        ]
    )
    return metric.measure(test_case, _show_indicator=False)


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("as_schema", [False, True])
@pytest.mark.parametrize(
    "mutate",
    [
        lambda name: name,
        lambda name: name.lower(),
        lambda name: f"{name}.",
        lambda name: f"${name}$",
    ],
    ids=["exact", "lowercase", "trailing-period", "dollar-wrapped"],
)
def test_measure_survives_winner_drift(mutate, as_schema, async_mode):
    assert _measure(mutate, async_mode, as_schema) == WINNING_CONTESTANT


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize(
    "mutate",
    [lambda name: "Tie", lambda name: f"{name} is the winner"],
    ids=["tie", "sentence"],
)
def test_measure_raises_deepeval_error_on_unresolvable_winner(
    mutate, async_mode
):
    with pytest.raises(DeepEvalError, match="could not resolve a winner"):
        _measure(mutate, async_mode)
