import json
import re
from collections.abc import Callable

import pytest

from deepeval.errors import DeepEvalError
from deepeval.metrics.arena_g_eval import ArenaGEval
from deepeval.metrics.arena_g_eval.arena_g_eval import _resolve_masked_winner
from deepeval.metrics.arena_g_eval.schema import RewrittenReason, Steps, Winner
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import (
    ArenaTestCase,
    Contestant,
    LLMTestCase,
    SingleTurnParams,
)


class _WinnerJudge(DeepEvalBaseLLM):
    def __init__(
        self,
        mutate: Callable[[str], str] | None = None,
        *,
        use_schema: bool = False,
        raw_payload: dict | None = None,
    ):
        self.mutate = mutate or (lambda name: name)
        self.use_schema = use_schema
        self.raw_payload = raw_payload

    def load_model(self, *args, **kwargs):
        return None

    def generate(self, prompt, schema=None, **kwargs):
        name = schema.__name__ if schema is not None else ""
        if name == "Steps":
            return (
                Steps(steps=["Step one: judge the outputs."])
                if self.use_schema
                else json.dumps({"steps": ["Step one: judge the outputs."]})
            )
        if name == "Winner":
            if self.raw_payload is not None:
                return (
                    self.raw_payload
                    if self.use_schema
                    else json.dumps(self.raw_payload)
                )
            match = re.search(r'"arena_test_cases":\s*{\s*"([^"]+)"', prompt)
            answer = self.mutate(match.group(1) if match else "unknown")
            return (
                Winner(winner=answer, reason="stub")
                if self.use_schema
                else json.dumps({"winner": answer, "reason": "stub"})
            )
        if name == "RewrittenReason":
            return (
                RewrittenReason(rewritten_reason="stub")
                if self.use_schema
                else json.dumps({"rewritten_reason": "stub"})
            )
        return json.dumps({"reason": "stub"})

    async def a_generate(self, prompt, schema=None, **kwargs):
        return self.generate(prompt, schema=schema, **kwargs)

    def get_model_name(self, *args, **kwargs):
        return "winner-judge"


def _make_arena() -> ArenaTestCase:
    return ArenaTestCase(
        contestants=[
            Contestant(
                name="model-alpha",
                test_case=LLMTestCase(input="hi", actual_output="Hello!"),
            ),
            Contestant(
                name="model-beta",
                test_case=LLMTestCase(input="hi", actual_output="Hi there."),
            ),
        ]
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda n: n,
        lambda n: n.lower(),
        lambda n: n.upper(),
        lambda n: f"  {n}.  ",
        lambda n: f'"{n}"',
        lambda n: f"${n}$",
        lambda n: f"{n} is the winner",
        lambda n: f"Winner: {n}!",
    ],
    ids=[
        "exact",
        "lower",
        "upper",
        "period_whitespace",
        "quotes",
        "dollar_wrapper",
        "sentence",
        "label_bang",
    ],
)
@pytest.mark.parametrize("use_schema", [False, True])
@pytest.mark.parametrize("async_mode", [False, True])
def test_arena_geval_normalizes_recoverable_winner_variants(
    mutate, use_schema, async_mode
):
    arena = _make_arena()
    metric = ArenaGEval(
        name="arena",
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
        ],
        criteria="Which reply is more helpful?",
        model=_WinnerJudge(mutate, use_schema=use_schema),
        async_mode=async_mode,
    )

    winner = metric.measure(arena, _show_indicator=False)

    assert winner in {"model-alpha", "model-beta"}
    assert metric.winner == winner
    assert metric.success is True


@pytest.mark.parametrize("bad_winner", ["Tie", "Both", "", "   ", "unknown"])
@pytest.mark.parametrize("async_mode", [False, True])
def test_arena_geval_invalid_winner_raises_deepeval_error(
    bad_winner, async_mode
):
    arena = _make_arena()
    metric = ArenaGEval(
        name="arena",
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
        ],
        criteria="Which reply is more helpful?",
        model=_WinnerJudge(lambda _n: bad_winner),
        async_mode=async_mode,
    )

    with pytest.raises(DeepEvalError, match="invalid winner"):
        metric.measure(arena, _show_indicator=False)

    assert metric.success is False
    assert metric.error is not None
    assert repr(bad_winner) in metric.error


def test_arena_geval_missing_winner_key_raises_deepeval_error():
    arena = _make_arena()
    metric = ArenaGEval(
        name="arena",
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
        ],
        criteria="Which reply is more helpful?",
        model=_WinnerJudge(raw_payload={"reason": "stub"}),
        async_mode=False,
    )

    with pytest.raises(DeepEvalError, match="invalid winner"):
        metric.measure(arena, _show_indicator=False)


def test_resolve_masked_winner_rejects_ambiguous_multi_name_answer():
    mapping = {"Alice": "model-a", "Bob": "model-b"}
    assert _resolve_masked_winner("Alice and Bob tie", mapping) is None


def test_resolve_masked_winner_respects_numeric_word_boundaries():
    mapping = {"Contestant1": "model-1", "Contestant10": "model-10"}
    assert (
        _resolve_masked_winner("Contestant10 is better", mapping)
        == "Contestant10"
    )
    assert (
        _resolve_masked_winner("Contestant1 is better", mapping)
        == "Contestant1"
    )
