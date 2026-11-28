import pytest

import deepeval.red_team.orchestration.sequential_agent as sequential_module
from deepeval.red_team.orchestration import SequentialAgent
from deepeval.red_team.orchestration.schema import SequentialVerdict, NextTurn
from deepeval.red_team.test_case.test_case import RTTurn


def verdict(met, focus, strategy="none", n=3):
    return SequentialVerdict(
        reason="r",
        scores=[5] * n,
        met=met,
        focus=focus,
        improvement_guidance="g",
        strategy_name=strategy,
        effectiveness="medium",
    )


class FakeGen:
    def __init__(self, verdicts):
        self.verdicts = list(verdicts)
        self.messages = 0
        self.judged = 0
        self.judge_prompts = []

    async def __call__(self, prompt, schema, model=None):
        if schema is NextTurn:
            self.messages += 1
            return NextTurn(message=f"message-{self.messages}")
        self.judge_prompts.append(prompt)
        self.judged += 1
        return self.verdicts[min(self.judged - 1, len(self.verdicts) - 1)]


@pytest.fixture
def target():
    def callback(*args):
        assert len(args) == 1  # memory-owning callback: no turns passed
        callback.calls.append(args[0])
        return RTTurn(role="assistant", content=f"reply-{len(callback.calls)}")

    callback.calls = []
    return callback


@pytest.fixture
def patch_gen(monkeypatch):
    def apply(fake):
        monkeypatch.setattr(sequential_module, "a_generate", fake)
        monkeypatch.setattr(
            sequential_module,
            "initialize_model",
            lambda model=None: (f"model:{model}", True),
        )

    return apply


def test_stops_as_soon_as_every_criteria_is_met(patch_gen, target):
    patch_gen(FakeGen([verdict([], 1), verdict([1], 2), verdict([1, 2, 3], 1)]))
    result = SequentialAgent(
        "t", "s", ["c1", "c2", "c3"], target, max_turns=10
    ).run()
    assert result.met_criteria == ["c1", "c2", "c3"]
    assert len(result.verdicts) == 3
    assert len(result.turns) == 6
    assert len(target.calls) == 3


def test_runs_to_the_budget_when_nothing_is_met(patch_gen, target):
    patch_gen(FakeGen([verdict([], 1, n=1)]))
    result = SequentialAgent("t", "s", ["c1"], target, max_turns=4).run()
    assert result.met_criteria == []
    assert len(result.verdicts) == 4
    assert len(target.calls) == 4


def test_score_is_the_fraction_of_criteria_met(patch_gen, target):
    patch_gen(FakeGen([verdict([1], 2), verdict([], 3)]))
    result = SequentialAgent(
        "t", "s", ["c1", "c2", "c3"], target, max_turns=2
    ).run()
    assert result.criteria == ["c1", "c2", "c3"]
    assert result.met_criteria == ["c1"]
    assert result.score == pytest.approx(1 / 3)


def test_score_is_one_when_all_met(patch_gen, target):
    patch_gen(FakeGen([verdict([1, 2], 1)]))
    result = SequentialAgent("t", "s", ["c1", "c2"], target, max_turns=5).run()
    assert result.score == 1.0


def test_met_criteria_accumulate_across_turns(patch_gen, target):
    patch_gen(
        FakeGen(
            [
                verdict([1], 2),
                verdict([2], 3),
                verdict([3], 1),
                verdict([], 1),
            ]
        )
    )
    result = SequentialAgent(
        "t", "s", ["c1", "c2", "c3"], target, max_turns=10
    ).run()
    assert sorted(result.met_criteria) == ["c1", "c2", "c3"]
    assert len(result.verdicts) == 3


def test_no_strategy_during_the_open_phase_then_adopts_it(patch_gen, target):
    patch_gen(FakeGen([verdict([], 1, strategy="Base64", n=1)]))
    SequentialAgent("t", "s", ["c1"], target, max_turns=6).run()
    assert all("Reply in Base64" not in c for c in target.calls[:3])
    assert "Reply in Base64" in target.calls[3]


def test_default_max_turns_is_thirty(target):
    assert SequentialAgent("t", "s", ["c1"], target).max_turns == 30


def test_strategy_cooldown(target):
    agent = SequentialAgent("t", "s", ["c1"], target)
    assert not agent._may_escalate(3)
    assert agent._may_escalate(4)
    agent.last_strategy_turn = 4
    assert not agent._may_escalate(5)
    assert not agent._may_escalate(6)
    assert agent._may_escalate(7)


def test_meeting_a_criteria_resets_the_strategy_state(target):
    agent = SequentialAgent("t", "s", ["c1", "c2"], target)
    agent.executor.adopt_by_name("Base64", "high")
    agent.last_strategy_turn = 4
    agent._reset_after_met(10)
    assert agent.executor.used == []
    assert agent.executor.strategy is None
    assert agent.last_strategy_turn == 10


def test_used_strategies_clear_when_a_criteria_is_met(patch_gen, target):
    patch_gen(
        FakeGen(
            [verdict([], 1, strategy="Base64", n=2)] * 4
            + [verdict([1], 1, n=2), verdict([1, 2], 1, n=2)]
        )
    )
    agent = SequentialAgent("t", "s", ["c1", "c2"], target, max_turns=8)
    agent.run()
    assert agent.executor.used == []


def test_judge_inventory_drops_an_already_used_strategy(patch_gen, target):
    fake = FakeGen([verdict([], 1, strategy="Base64", n=1)])
    patch_gen(fake)
    SequentialAgent("t", "s", ["c1"], target, max_turns=6).run()

    before = fake.judge_prompts[0]
    after = fake.judge_prompts[3]
    base64_before = next(
        line for line in before.splitlines() if line.startswith("- Base64")
    )
    base64_after = next(
        line for line in after.splitlines() if line.startswith("- Base64")
    )
    assert "low, medium, high" in base64_before
    assert "medium" not in base64_after
    assert "low, high" in base64_after


def test_encoded_reply_is_decoded_for_the_record_and_the_judge(patch_gen):
    import base64

    def target(*args):
        if "Reply in Base64" in args[0]:
            return RTTurn(
                role="assistant",
                content=base64.b64encode(b"a lion").decode(),
            )
        return RTTurn(role="assistant", content="plain")

    fake = FakeGen([verdict([], 1, strategy="Base64", n=1)])
    patch_gen(fake)
    result = SequentialAgent("t", "s", ["c1"], target, max_turns=6).run()

    assert any(
        t.role == "assistant" and t.content == "a lion" for t in result.turns
    )
    assert "a lion" in fake.judge_prompts[3]


def test_an_undecodable_reply_is_kept_raw(patch_gen):
    def target(*args):
        return RTTurn(role="assistant", content="I will not comply")

    patch_gen(FakeGen([verdict([], 1, strategy="Base64", n=1)]))
    result = SequentialAgent("t", "s", ["c1"], target, max_turns=6).run()
    assert any(
        t.role == "assistant" and t.content == "I will not comply"
        for t in result.turns
    )


def test_a_target_error_ends_the_run_gracefully(patch_gen):
    patch_gen(FakeGen([verdict([], 1, n=1)]))
    state = {"calls": 0}

    def flaky(*args):
        state["calls"] += 1
        if state["calls"] == 2:
            raise RuntimeError("target exploded")
        return RTTurn(role="assistant", content="ok")

    result = SequentialAgent("t", "s", ["c1"], flaky, max_turns=5).run()
    assert isinstance(result.error, RuntimeError)
    assert "target exploded" in str(result.error)
    assert result.met_criteria == []
    assert len(result.turns) == 2


def test_rejects_empty_criteria(target):
    with pytest.raises(ValueError):
        SequentialAgent("t", "s", [], target)


def test_a_callback_returning_a_string_is_accepted(patch_gen):
    patch_gen(FakeGen([verdict([1], 1, n=1)]))

    def callback(message):
        return f"plain reply to {message}"

    result = SequentialAgent("t", "s", ["c1"], callback, max_turns=1).run()
    assert result.error is None
    assert result.turns[1] == RTTurn(
        role="assistant", content="plain reply to message-1"
    )
