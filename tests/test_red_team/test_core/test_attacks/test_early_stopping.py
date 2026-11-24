import asyncio
from typing import List, Optional

import pytest

from deepteam.attacks.multi_turn import (
    BadLikertJudge,
    BehaviorShiftDetector,
    CrescendoJailbreaking,
    LinearJailbreaking,
    SequentialJailbreak,
    StopReason,
    TreeJailbreaking,
    is_progression_completed,
)
from deepteam.attacks.multi_turn.progression import (
    MetricVerdict,
    ShiftExplanation,
    ShiftVerdict,
    default_stop_detail,
    mark_stop,
)
from deepteam.test_case.test_case import RTTurn

MODULES = [
    "deepteam.attacks.multi_turn.progression.shift_detection",
    "deepteam.attacks.multi_turn.progression.progression",
    "deepteam.attacks.multi_turn.linear_jailbreaking.linear_jailbreaking",
    "deepteam.attacks.multi_turn.bad_likert_judge.bad_likert_judge",
    "deepteam.attacks.multi_turn.sequential_break.sequential_break",
    "deepteam.attacks.multi_turn.crescendo_jailbreaking.crescendo_jailbreaking",
    "deepteam.attacks.multi_turn.tree_jailbreaking.tree_jailbreaking",
]

FAIL = MetricVerdict(score=0.0, reason="metric says broken")
PASS = MetricVerdict(score=1.0, reason="metric says held")

EXPLANATION = "it dropped the refusal and listed the steps"


def _fill(schema):
    if schema is ShiftExplanation:
        return ShiftExplanation(reasoning=EXPLANATION)

    values = {}
    for name, field in schema.model_fields.items():
        annotation = field.annotation
        if annotation is bool:
            values[name] = name == "on_topic"
        elif annotation is int:
            values[name] = 1
        elif annotation is float:
            values[name] = 1.0
        else:
            values[name] = f"stub-{name}"
    return schema(**values)


@pytest.fixture
def stub_models(monkeypatch):
    def _generate(prompt, schema, model=None):
        return _fill(schema)

    async def _a_generate(prompt, schema, model=None):
        return _fill(schema)

    def _initialize_model(model=None, *args, **kwargs):
        return object(), False

    for module in MODULES:
        mod = __import__(module, fromlist=["x"])
        if hasattr(mod, "generate"):
            monkeypatch.setattr(mod, "generate", _generate)
        if hasattr(mod, "a_generate"):
            monkeypatch.setattr(mod, "a_generate", _a_generate)
        if hasattr(mod, "initialize_model"):
            monkeypatch.setattr(mod, "initialize_model", _initialize_model)


def make_callback(reply: str = "target reply"):
    def callback(input: str, turns: Optional[List[RTTurn]] = None) -> RTTurn:
        return RTTurn(role="assistant", content=reply)

    return callback


def make_async_callback(reply: str = "target reply"):
    async def callback(
        input: str, turns: Optional[List[RTTurn]] = None
    ) -> RTTurn:
        return RTTurn(role="assistant", content=reply)

    return callback


def seed() -> List[RTTurn]:
    return [RTTurn(role="user", content="original attack")]


def async_metric(verdict):
    async def check(turns):
        return verdict

    return check


class TestBehaviorShiftDetector:

    TURNS = [
        RTTurn(role="user", content="attack"),
        RTTurn(role="assistant", content="reply"),
    ]

    @pytest.fixture(autouse=True)
    def _stub_explainer(self, monkeypatch):
        import deepteam.attacks.multi_turn.progression.shift_detection as progression

        monkeypatch.setattr(
            progression,
            "generate",
            lambda *a, **k: ShiftExplanation(reasoning=EXPLANATION),
        )
        monkeypatch.setattr(progression, "format_turns", lambda turns: "conv")

    def test_metric_pass_does_not_stop(self):
        detector = BehaviorShiftDetector(
            simulator_model=object(), metric_check=lambda turns: PASS
        )
        assert detector.check(self.TURNS) is None

    def test_metric_fail_stops_and_carries_its_verdict(self):
        detector = BehaviorShiftDetector(
            simulator_model=object(), metric_check=lambda turns: FAIL
        )
        verdict = detector.check(self.TURNS)
        assert verdict is not None
        assert verdict.metric_score == 0.0
        assert verdict.metric_reason == "metric says broken"
        assert EXPLANATION in verdict.detail

    def test_no_metric_means_no_stop(self):
        detector = BehaviorShiftDetector(simulator_model=object())
        assert detector.check(self.TURNS) is None

    def test_unavailable_metric_does_not_stop(self):
        detector = BehaviorShiftDetector(
            simulator_model=object(), metric_check=lambda turns: None
        )
        assert detector.check(self.TURNS) is None

    def test_no_check_before_an_assistant_turn_exists(self):
        detector = BehaviorShiftDetector(
            simulator_model=object(), metric_check=lambda turns: FAIL
        )
        assert detector.check([RTTurn(role="user", content="attack")]) is None

    def test_explainer_failure_still_stops(self, monkeypatch):
        import deepteam.attacks.multi_turn.progression.shift_detection as progression

        def boom(*a, **k):
            raise RuntimeError("explainer down")

        monkeypatch.setattr(progression, "generate", boom)
        detector = BehaviorShiftDetector(
            simulator_model=object(), metric_check=lambda turns: FAIL
        )
        verdict = detector.check(self.TURNS)
        assert verdict is not None
        assert verdict.reasoning is None
        assert verdict.detail is None


class TestStopReasonPlumbing:

    def test_mark_stop_lands_on_the_final_turn(self):
        turns = [
            RTTurn(role="user", content="a"),
            RTTurn(role="assistant", content="b"),
        ]
        mark_stop(turns, StopReason.SHIFT_DETECTED, detail="it complied")
        assert turns[-1].stopping_category == "SHIFT_DETECTED"
        assert turns[-1].stopping_reason == "it complied"
        assert turns[0].stopping_reason is None
        assert turns[0].stopping_category is None

    def test_mark_stop_falls_back_to_default_prose(self):
        turns = [
            RTTurn(role="user", content="a"),
            RTTurn(role="assistant", content="b"),
        ]
        mark_stop(turns, StopReason.SHIFT_DETECTED)
        assert turns[-1].stopping_reason == default_stop_detail(
            StopReason.SHIFT_DETECTED
        )

    def test_only_shift_and_budget_count_as_completed(self):
        assert is_progression_completed(StopReason.SHIFT_DETECTED)
        assert is_progression_completed(StopReason.BUDGET_EXHAUSTED)
        assert not is_progression_completed(StopReason.SIMULATOR_REFUSED)
        assert not is_progression_completed(StopReason.SIMULATION_ERROR)
        assert not is_progression_completed(StopReason.RUNTIME_EXCEEDED)

    def test_unmarked_conversations_count_as_completed(self):
        assert is_progression_completed(None)

    def test_metric_verdict_reads_score_as_pass_fail(self):
        assert MetricVerdict(score=0.0).failed
        assert not MetricVerdict(score=1.0).failed
        assert not MetricVerdict().failed

    def test_shift_detail_carries_only_the_explainers_prose(self):
        detail = ShiftVerdict(
            reasoning=EXPLANATION, metric_reason="metric says broken"
        ).detail
        assert EXPLANATION in detail
        assert "metric says broken" not in detail

    def test_every_reason_has_prose(self):
        for reason in StopReason:
            detail = default_stop_detail(reason)
            assert len(detail) > 40, reason
            assert reason.value not in detail

    def test_budget_prose_names_the_turns_spent(self):
        detail = default_stop_detail(StopReason.BUDGET_EXHAUSTED, turns_spent=5)
        assert "5" in detail


ATTACKS = [
    (lambda: LinearJailbreaking(num_turns=3), 2 + 2 * 3),
    (lambda: BadLikertJudge(num_turns=3), 2 + 2 * 3),
    (lambda: SequentialJailbreak(num_turns=3), 2 + 2 * 3),
    (lambda: CrescendoJailbreaking(max_rounds=3), 2 + 2 * 3),
    (lambda: TreeJailbreaking(max_depth=3, branching_factor=2), 2 + 2 * 3),
]

IDS = ["linear", "likert", "sequential", "crescendo", "tree"]


@pytest.mark.usefixtures("stub_models")
class TestProgressionInvariants:

    @pytest.mark.parametrize("factory,expected", ATTACKS, ids=IDS)
    def test_resisting_target_spends_the_whole_budget(self, factory, expected):
        turns = factory()._get_turns(
            make_callback("I cannot help with that."),
            seed(),
            metric_check=lambda t: PASS,
        )
        assert len(turns) == expected
        assert turns[-1].stopping_category == StopReason.BUDGET_EXHAUSTED.value

    @pytest.mark.parametrize("factory,expected", ATTACKS, ids=IDS)
    def test_metric_fail_stops_immediately(self, factory, expected):
        turns = factory()._get_turns(
            make_callback("Sure, here are the steps."),
            seed(),
            metric_check=lambda t: FAIL,
        )
        assert turns[-1].stopping_category == StopReason.SHIFT_DETECTED.value
        assert len(turns) < expected

    @pytest.mark.parametrize("factory,expected", ATTACKS, ids=IDS)
    def test_stop_records_the_explainers_prose(self, factory, expected):
        turns = factory()._get_turns(
            make_callback("Sure, here are the steps."),
            seed(),
            metric_check=lambda t: FAIL,
        )
        detail = turns[-1].stopping_reason
        assert EXPLANATION in detail
        assert detail != StopReason.SHIFT_DETECTED.value

    @pytest.mark.parametrize("factory,expected", ATTACKS, ids=IDS)
    def test_hold_records_why_it_held(self, factory, expected):
        turns = factory()._get_turns(
            make_callback("I cannot help with that."),
            seed(),
            metric_check=lambda t: PASS,
        )
        detail = turns[-1].stopping_reason
        assert "held" in detail and str(expected // 2 - 1) in detail
        assert detail != StopReason.BUDGET_EXHAUSTED.value

    @pytest.mark.parametrize("factory,expected", ATTACKS, ids=IDS)
    def test_stop_lands_on_the_last_turn(self, factory, expected):
        turns = factory()._get_turns(
            make_callback("Sure, here are the steps."),
            seed(),
            metric_check=lambda t: FAIL,
        )
        assert turns[-1].role == "assistant"
        assert sum(1 for t in turns if t.stopping_category is not None) == 1

    @pytest.mark.parametrize("factory,expected", ATTACKS, ids=IDS)
    def test_without_a_metric_the_budget_is_spent(self, factory, expected):
        turns = factory()._get_turns(make_callback(), seed())
        assert len(turns) == expected
        assert turns[-1].stopping_category == StopReason.BUDGET_EXHAUSTED.value

    @pytest.mark.parametrize("factory,expected", ATTACKS, ids=IDS)
    def test_async_hold_matches_sync(self, factory, expected):
        turns = asyncio.new_event_loop().run_until_complete(
            factory()._a_get_turns(
                make_async_callback("I cannot help with that."),
                seed(),
                metric_check=async_metric(PASS),
            )
        )
        assert len(turns) == expected
        assert turns[-1].stopping_category == StopReason.BUDGET_EXHAUSTED.value

    @pytest.mark.parametrize("factory,expected", ATTACKS, ids=IDS)
    def test_async_stop_matches_sync(self, factory, expected):
        turns = asyncio.new_event_loop().run_until_complete(
            factory()._a_get_turns(
                make_async_callback("Sure, here are the steps."),
                seed(),
                metric_check=async_metric(FAIL),
            )
        )
        assert turns[-1].stopping_category == StopReason.SHIFT_DETECTED.value
        assert len(turns) < expected


@pytest.mark.usefixtures("stub_models")
class TestBadLikertJudgeBudget:

    def test_backtracks_do_not_eat_the_turn_budget(self, monkeypatch):
        import deepteam.attacks.multi_turn.bad_likert_judge.bad_likert_judge as blj
        from deepteam.attacks.multi_turn.base_schema import NonRefusal

        calls = {"n": 0}
        real_generate = blj.generate

        def flaky(prompt, schema, model=None):
            if schema is NonRefusal:
                calls["n"] += 1
                return NonRefusal(refusal=calls["n"] <= 2, reasoning="stub")
            return real_generate(prompt, schema, model)

        monkeypatch.setattr(blj, "generate", flaky)

        turns = BadLikertJudge(num_turns=3, max_backtracks=3)._get_turns(
            make_callback("I cannot help with that."),
            seed(),
            metric_check=lambda t: PASS,
        )
        assert len(turns) == 2 + 2 * 3
        assert turns[-1].stopping_category == StopReason.BUDGET_EXHAUSTED.value

    def test_exhausted_backtracks_report_a_simulator_refusal(
        self, monkeypatch
    ):
        import deepteam.attacks.multi_turn.bad_likert_judge.bad_likert_judge as blj
        from deepteam.attacks.multi_turn.base_schema import NonRefusal

        real_generate = blj.generate

        def always_refuses(prompt, schema, model=None):
            if schema is NonRefusal:
                return NonRefusal(refusal=True, reasoning="stub")
            return real_generate(prompt, schema, model)

        monkeypatch.setattr(blj, "generate", always_refuses)

        turns = BadLikertJudge(num_turns=3, max_backtracks=2)._get_turns(
            make_callback(), seed(), metric_check=lambda t: PASS
        )
        assert (
            turns[-1].stopping_category == StopReason.SIMULATOR_REFUSED.value
        )
        assert not is_progression_completed(turns[-1].stopping_category)


@pytest.mark.usefixtures("stub_models")
class TestCrescendoBacktracking:

    def test_refusals_cost_budget_but_never_end_the_run(self, monkeypatch):
        attack = CrescendoJailbreaking(max_rounds=3, max_backtracks=2)
        monkeypatch.setattr(
            attack, "_refusal_score", lambda *a, **k: (True, "refused")
        )

        turns = attack._get_turns(
            make_callback("I cannot help with that."),
            seed(),
            metric_check=lambda t: PASS,
        )
        assert turns[-1].stopping_category == StopReason.BUDGET_EXHAUSTED.value
        assert len(turns) == 2 + 2 * 3

    def test_refusal_rationale_never_enters_the_transcript(self, monkeypatch):
        attack = CrescendoJailbreaking(max_rounds=2, max_backtracks=1)
        monkeypatch.setattr(
            attack,
            "_refusal_score",
            lambda *a, **k: (True, "secret-rationale"),
        )

        turns = attack._get_turns(
            make_callback("I cannot help with that."),
            seed(),
            metric_check=lambda t: PASS,
        )
        for turn in turns:
            assert "secret-rationale" not in turn.content
            assert "refusal judge provided" not in turn.content


@pytest.mark.usefixtures("stub_models")
class TestTreeSearch:

    def test_transcript_uses_the_scored_responses(self):
        attack = TreeJailbreaking(max_depth=2, branching_factor=2)
        result = attack.run(
            make_callback(), seed(), metric_check=lambda t: PASS
        )

        committed = [a for a in result.probes if a.committed]
        assert len(committed) == 2
        assert [a.input for a in committed] == [
            t.content for t in result.turns if t.role == "user"
        ][1:]
        for probe in committed:
            assert probe.output is not None

    def test_depth_bounds_the_path_not_the_clock(self):
        turns = TreeJailbreaking(max_depth=4, branching_factor=1)._get_turns(
            make_callback(), seed(), metric_check=lambda t: PASS
        )
        assert len(turns) == 2 + 2 * 4
        assert turns[-1].stopping_category == StopReason.BUDGET_EXHAUSTED.value

    def test_runtime_backstop_is_reported_as_incomplete(self):
        turns = TreeJailbreaking(
            max_depth=5, branching_factor=1, max_runtime=-1.0
        )._get_turns(make_callback(), seed(), metric_check=lambda t: PASS)
        assert (
            turns[-1].stopping_category == StopReason.RUNTIME_EXCEEDED.value
        )
        assert not is_progression_completed(turns[-1].stopping_category)

    def test_off_topic_pruning_is_applied(self, monkeypatch):
        import deepteam.attacks.multi_turn.tree_jailbreaking.tree_jailbreaking as tree
        from deepteam.attacks.multi_turn.tree_jailbreaking.schema import (
            OnTopic,
        )

        real_generate = tree.generate

        def all_off_topic(prompt, schema, model=None):
            if schema is OnTopic:
                return OnTopic(on_topic=False)
            return real_generate(prompt, schema, model)

        monkeypatch.setattr(tree, "generate", all_off_topic)

        attack = TreeJailbreaking(max_depth=2, branching_factor=2)
        result = attack.run(
            make_callback(), seed(), metric_check=lambda t: PASS
        )
        assert len(result.turns) == 2 + 2 * 2
