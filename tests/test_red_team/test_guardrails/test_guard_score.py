"""Key-free regression tests for `BaseGuard` score accounting.

`SafetyLevelSchema.safety_level` is a `Literal["safe", "borderline", "unsafe"]`,
so every guard path stores a plain string on `BaseGuard.safety_level`. The score
was computed by comparing that string against `SafetyLevel.SAFE`, which is an
`Enum` member rather than a `str` member, so the comparison could never hold and
`score` was pinned at 0.0 for safe content (and 0.0 was then reported through the
documented `GuardVerdict.score` field).

See also `GuardResult._normalize_level`, which already accepts either an enum or
a raw string, and `tests/test_guardrails/*`, which already assert the guards
return plain strings such as `"safe"`.
"""

import asyncio

import pytest

from deepeval.models import DeepEvalBaseLLM
from deepeval.red_team.guardrails import Guardrails
from deepeval.red_team.guardrails.guards import ToxicityGuard
from deepeval.red_team.guardrails.guards.schema import SafetyLevelSchema


class _StubJudge(DeepEvalBaseLLM):
    """Deterministic stand-in for a custom model. Makes no network calls."""

    def __init__(self, safety_level: str = "safe"):
        self.stub_safety_level = safety_level
        super().__init__("stub-judge")

    def load_model(self, *args, **kwargs):
        return self

    def generate(self, *args, **kwargs):
        if kwargs.get("schema") is not None:
            return SafetyLevelSchema(
                safety_level=self.stub_safety_level, reason="stubbed verdict"
            )
        raise TypeError("this stub only answers schema-guided requests")

    async def a_generate(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    def get_model_name(self, *args, **kwargs):
        return "stub-judge"


@pytest.fixture(autouse=True)
def no_telemetry(monkeypatch):
    monkeypatch.setattr(
        "deepeval.red_team.guardrails.guards.base_guard.record_guard",
        lambda *args, **kwargs: None,
    )


@pytest.mark.parametrize(
    "safety_level, expected_score",
    [("safe", 1.0), ("borderline", 0.0), ("unsafe", 0.0)],
)
def test_sync_guard_score_follows_safety_level(safety_level, expected_score):
    guard = ToxicityGuard(model=_StubJudge(safety_level))

    assert guard.guard_input("What is the weather like today?") == safety_level
    assert guard.score == expected_score


@pytest.mark.parametrize(
    "safety_level, expected_score",
    [("safe", 1.0), ("borderline", 0.0), ("unsafe", 0.0)],
)
def test_async_guard_score_follows_safety_level(safety_level, expected_score):
    guard = ToxicityGuard(model=_StubJudge(safety_level))

    asyncio.run(guard.a_guard_input("What is the weather like today?"))
    assert guard.score == expected_score


def test_guard_verdict_score_is_reported_for_safe_input():
    judge = _StubJudge("safe")
    guardrails = Guardrails(
        input_guards=[ToxicityGuard(model=judge)],
        output_guards=[ToxicityGuard(model=judge)],
        evaluation_model=judge,
    )

    result = guardrails.guard_input("What is the weather like today?")

    assert result.breached is False
    assert len(result.verdicts) == 1
    verdict = result.verdicts[0]
    assert verdict.safety_level == "safe"
    assert verdict.error is None
    assert verdict.score == 1.0


class _NativeJudge(_StubJudge):
    """Mimics a native model: `generate` returns the parsed schema plus a cost."""

    def generate(self, *args, **kwargs):
        return (
            SafetyLevelSchema(
                safety_level=self.stub_safety_level, reason="stubbed verdict"
            ),
            0.0,
        )

    async def a_generate(self, *args, **kwargs):
        return self.generate(*args, **kwargs)


def test_native_guard_path_score(monkeypatch):
    monkeypatch.setattr(
        "deepeval.red_team.guardrails.guards.base_guard.initialize_model",
        lambda model: (_NativeJudge("safe"), True),
    )
    guard = ToxicityGuard(model="stub-native")

    assert guard.guard_input("What is the weather like today?") == "safe"
    assert guard.score == 1.0


class _JsonOnlyJudge(DeepEvalBaseLLM):
    """An older custom model whose `generate` takes no `schema` argument, which is
    what routes the guard through the raw-JSON fallback branch."""

    def __init__(self):
        super().__init__("stub-json-judge")

    def load_model(self, *args, **kwargs):
        return self

    def generate(self, prompt):
        return '{"safety_level": "safe", "reason": "stubbed verdict"}'

    async def a_generate(self, prompt):
        return self.generate(prompt)

    def get_model_name(self, *args, **kwargs):
        return "stub-json-judge"


def test_json_fallback_guard_path_score():
    guard = ToxicityGuard(model=_JsonOnlyJudge())

    assert guard.guard_input("What is the weather like today?") == "safe"
    assert guard.score == 1.0
