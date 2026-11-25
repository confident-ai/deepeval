import asyncio

import pytest

from deepeval.red_team.attacks.multi_turn.progression import Probe
from deepeval.red_team.orchestration.strategy import (
    AntiRefusalStrategy,
    Base64Strategy,
    Effectiveness,
    GradualEscalationStrategy,
    Strategy,
    TemplateStrategy,
    StrategyExecutor,
    anti_refusal,
    build_strategy,
    strategy_catalog,
)
from deepeval.red_team.test_case.test_case import RTTurn


def run(coro):
    return asyncio.run(coro)


class RecordingSend:

    def __init__(self):
        self.sent = []

    async def __call__(self, text: str) -> Probe:
        probe = Probe(
            id=len(self.sent),
            input=text,
            output=RTTurn(role="assistant", content=f"reply-to-{text}"),
        )
        self.sent.append(probe)
        return probe


class Suffix(Strategy):
    name = "suffix"
    description = "d"
    strategy = "s"
    effectiveness = Effectiveness.LOW

    async def a_execute(self, message, send):
        await send(f"{message} [suffix]")


class Priming(Strategy):
    name = "priming"
    description = "d"
    strategy = "s"
    effectiveness = Effectiveness.MEDIUM

    async def a_execute(self, message, send):
        for i in range(2):
            await send(f"shot-{i}")
        await send(message)


class Silent(Strategy):
    name = "silent"
    description = "d"
    strategy = "s"
    effectiveness = Effectiveness.LOW

    async def a_execute(self, message, send):
        return


def executor():
    return StrategyExecutor(sub_agent=None)


class TestPassThrough:

    def test_no_strategy_sends_the_message_unchanged(self):
        send = RecordingSend()
        probe = run(executor().a_send("hello", send))
        assert [p.input for p in send.sent] == ["hello"]
        assert probe is send.sent[-1]


class TestSingleTurnStrategy:

    def test_strategy_rewrites_the_one_delivered_message(self):
        send = RecordingSend()
        ex = executor()
        ex.adopt(Suffix())
        probe = run(ex.a_send("hello", send))
        assert [p.input for p in send.sent] == ["hello [suffix]"]
        assert probe is send.sent[-1]


class TestMultiTurnStrategy:

    def test_setup_turns_precede_the_real_message(self):
        send = RecordingSend()
        ex = executor()
        ex.adopt(Priming())
        probe = run(ex.a_send("the-real-message", send))
        assert [p.input for p in send.sent] == [
            "shot-0",
            "shot-1",
            "the-real-message",
        ]
        assert probe is send.sent[-1]
        assert probe.input == "the-real-message"


class TestDegenerateStrategy:

    def test_a_strategy_that_sends_nothing_still_delivers_the_message(self):
        send = RecordingSend()
        ex = executor()
        ex.adopt(Silent())
        probe = run(ex.a_send("hello", send))
        assert [p.input for p in send.sent] == ["hello"]
        assert probe is send.sent[-1]


class TestStrategyContract:

    def test_decode_defaults_to_none(self):
        assert Suffix().decode("anything") is None

    def test_adopt_sets_the_strategy(self):
        ex = executor()
        strategy = Suffix()
        ex.adopt(strategy)
        assert ex.strategy is strategy


class Demo(TemplateStrategy):
    name = "demo"
    description = "d"
    strategy = "s"
    LOW_TEMPLATE = "LOW: {message}"
    MEDIUM_TEMPLATE = "{message} [MED]"
    HIGH_TEMPLATE = "<<{message}>>"


class Empty(TemplateStrategy):
    name = "empty"
    description = "d"
    strategy = "s"


class Braces(TemplateStrategy):
    name = "braces"
    description = "d"
    strategy = "s"
    MEDIUM_TEMPLATE = '{"role": "admin"}\n{message}'


class TestTemplateStrategy:

    def test_fills_the_template_by_effectiveness(self):
        for effectiveness, expected in [
            (Effectiveness.LOW, "LOW: hello"),
            (Effectiveness.MEDIUM, "hello [MED]"),
            (Effectiveness.HIGH, "<<hello>>"),
        ]:
            send = RecordingSend()
            ex = executor()
            ex.adopt(Demo(effectiveness=effectiveness))
            run(ex.a_send("hello", send))
            assert send.sent[-1].input == expected

    def test_the_default_template_sends_the_message_unchanged(self):
        send = RecordingSend()
        ex = executor()
        ex.adopt(Empty())
        run(ex.a_send("hello", send))
        assert send.sent[-1].input == "hello"

    def test_literal_braces_in_a_template_are_preserved(self):
        send = RecordingSend()
        ex = executor()
        ex.adopt(Braces())
        run(ex.a_send("hi", send))
        assert send.sent[-1].input == '{"role": "admin"}\nhi'


class TestGradualEscalation:

    def test_warms_up_then_asks_the_real_question(self):
        strategy = GradualEscalationStrategy(effectiveness=Effectiveness.HIGH)

        async def generate(prompt):
            generate.n += 1
            return f"warmup-{generate.n}"

        generate.n = 0
        strategy.generate = generate
        send = RecordingSend()
        run(strategy.a_execute("THE REAL QUESTION", send))

        inputs = [p.input for p in send.sent]
        assert inputs[:3] == ["warmup-1", "warmup-2", "warmup-3"]
        assert (
            "THE REAL QUESTION" in inputs[3]
            and inputs[3] != "THE REAL QUESTION"
        )

    def test_warmup_count_scales_with_effectiveness(self):
        async def generate(prompt):
            return "w"

        for effectiveness, warmups in [
            (Effectiveness.LOW, 1),
            (Effectiveness.MEDIUM, 2),
            (Effectiveness.HIGH, 3),
        ]:
            strategy = GradualEscalationStrategy(effectiveness=effectiveness)
            strategy.generate = generate
            send = RecordingSend()
            run(strategy.a_execute("q", send))
            assert len(send.sent) == warmups + 1

    def test_warms_up_only_once_per_adoption(self):
        async def generate(prompt):
            return "w"

        strategy = GradualEscalationStrategy(effectiveness=Effectiveness.HIGH)
        strategy.generate = generate
        send = RecordingSend()
        run(strategy.a_execute("first", send))
        run(strategy.a_execute("second", send))

        assert len(send.sent) == 3 + 1 + 1
        assert send.sent[-1].input == "second"

    def test_without_a_generator_it_only_sends_the_wrapped_question(self):
        strategy = GradualEscalationStrategy(effectiveness=Effectiveness.HIGH)
        send = RecordingSend()
        run(strategy.a_execute("q", send))
        assert len(send.sent) == 1
        assert "q" in send.sent[-1].input

    def test_the_executor_injects_a_generator_on_adopt(self):
        ex = StrategyExecutor(sub_agent=None, simulator_model="model")
        strategy = GradualEscalationStrategy()
        ex.adopt(strategy)
        assert callable(strategy.generate)


class TestRegistry:

    def test_build_strategy_by_name_and_effectiveness(self):
        strategy = build_strategy("Base64", "high")
        assert isinstance(strategy, Base64Strategy)
        assert strategy.effectiveness is Effectiveness.HIGH

    def test_build_unknown_name_returns_none(self):
        assert build_strategy("Nonexistent", "high") is None
        assert build_strategy("none", "high") is None

    def test_catalog_lists_every_registered_strategy(self):
        names = {entry["name"] for entry in strategy_catalog()}
        assert {"Anti-refusal", "Base64", "ROT-13", "Leetspeak"} <= names
        assert all(entry["description"] for entry in strategy_catalog())
        for entry in strategy_catalog():
            assert entry["effectiveness"] == ["low", "medium", "high"]

    def test_catalog_drops_an_excluded_pair(self):
        catalog = strategy_catalog(exclude=[("Base64", "medium")])
        base64 = next(e for e in catalog if e["name"] == "Base64")
        assert base64["effectiveness"] == ["low", "high"]

    def test_catalog_drops_a_strategy_with_no_levels_left(self):
        catalog = strategy_catalog(
            exclude=[("Base64", lvl) for lvl in ("low", "medium", "high")]
        )
        assert all(entry["name"] != "Base64" for entry in catalog)


class TestExecutorAdoptByName:

    def test_adopt_by_name_picks_from_the_registry(self):
        ex = executor()
        ex.adopt_by_name("Base64", "medium")
        assert isinstance(ex.strategy, Base64Strategy)

    def test_adopt_by_name_ignores_an_unknown_strategy(self):
        ex = executor()
        ex.adopt_by_name("Nonexistent", "medium")
        assert ex.strategy is None


class TestStrategyTracking:

    def test_adopted_strategies_are_tracked_by_name_and_effectiveness(self):
        ex = executor()
        ex.adopt_by_name("Base64", "high")
        ex.adopt_by_name("Anti-refusal", "medium")
        assert ex.used == [("Base64", "high"), ("Anti-refusal", "medium")]
        assert ex.has_used("Base64", "high")
        assert not ex.has_used("Base64", "low")
        assert not ex.has_used("ROT-13", "high")

    def test_effectiveness_is_normalised_from_the_built_strategy(self):
        ex = executor()
        ex.adopt_by_name("Base64", "nonsense")
        assert ex.used == [("Base64", "medium")]

    def test_an_unknown_strategy_is_not_tracked(self):
        ex = executor()
        ex.adopt_by_name("Nonexistent", "high")
        assert ex.used == []


class TestOrchestratorWiring:

    def test_create_sub_agent_adopts_the_named_strategy(self, make_agent):
        agent = make_agent()
        sub = agent.create_sub_agent(
            "t", "c1", "s", "a", strategy_name="Base64", effectiveness="high"
        )
        assert isinstance(sub.executor.strategy, Base64Strategy)
        assert sub.executor.strategy.effectiveness is Effectiveness.HIGH

    def test_create_sub_agent_without_a_strategy_is_passthrough(
        self, make_agent
    ):
        agent = make_agent()
        sub = agent.create_sub_agent("t", "c1", "s", "a")
        assert sub.executor.strategy is None

    def test_sub_agents_start_without_a_strategy(self, make_agent, llm):
        agent = make_agent(criteria=["c1", "c2"], agents_per_criteria=2)

        async def go():
            analysis = await agent.a_recon()
            await agent.a_create_sub_agents(analysis)

        run(go())
        assert len(agent.round_sub_agents) == 4
        assert all(
            sub.executor.strategy is None for sub in agent.round_sub_agents
        )


class FakeDataset:

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def fake_alpaca(n):
    rows = [
        {"instruction": f"q{i}", "input": "", "output": f"a{i}"}
        for i in range(n)
    ]
    return lambda *args, **kwargs: FakeDataset(rows)


class TestAntiRefusalFewShot:

    @pytest.fixture(autouse=True)
    def fresh_examples(self, monkeypatch):
        monkeypatch.setattr(anti_refusal, "_EXAMPLES", None)

    def test_batches_are_framed_then_the_real_question(self, monkeypatch):
        datasets = pytest.importorskip("datasets")

        monkeypatch.setattr(anti_refusal, "_SAMPLE_SIZE", 3)
        monkeypatch.setattr(anti_refusal, "_BATCH_SIZE", 2)
        monkeypatch.setattr(anti_refusal, "_ANSWER_PREFIXES", ["JARVIS:"])
        monkeypatch.setattr(datasets, "load_dataset", fake_alpaca(3))
        send = RecordingSend()
        ex = executor()
        ex.adopt(AntiRefusalStrategy(effectiveness=Effectiveness.HIGH))
        probe = run(ex.a_send("REAL?", send))

        texts = [p.input for p in send.sent]
        assert len(texts) == 3
        assert texts[0].startswith(anti_refusal._FEW_SHOT_INTRO)
        assert "Q: q" in texts[0] and "A: JARVIS: a" in texts[0]
        assert texts[1].startswith(anti_refusal._FEW_SHOT_MORE)
        assert texts[2].startswith(anti_refusal._FEW_SHOT_ASK)
        assert texts[2].endswith("REAL?")
        assert probe is send.sent[-1]

    def test_every_answer_gets_the_same_prefix(self, monkeypatch):
        datasets = pytest.importorskip("datasets")

        monkeypatch.setattr(anti_refusal, "_SAMPLE_SIZE", 4)
        monkeypatch.setattr(anti_refusal, "_BATCH_SIZE", 4)
        monkeypatch.setattr(anti_refusal, "_ANSWER_PREFIXES", ["AAA:", "BBB:"])
        monkeypatch.setattr(datasets, "load_dataset", fake_alpaca(4))
        send = RecordingSend()
        ex = executor()
        ex.adopt(AntiRefusalStrategy(effectiveness=Effectiveness.HIGH))
        run(ex.a_send("REAL?", send))

        answer_lines = [
            line
            for line in send.sent[0].input.splitlines()
            if line.startswith("A: ")
        ]
        assert len(answer_lines) == 4
        prefixes = {
            line[len("A: ") :].split(" ", 1)[0] for line in answer_lines
        }
        assert len(prefixes) == 1
        assert prefixes <= {"AAA:", "BBB:"}

    def test_only_high_effectiveness_runs_few_shot(self, monkeypatch):
        datasets = pytest.importorskip("datasets")

        def boom(*args, **kwargs):
            raise AssertionError("few-shot must run only at HIGH effectiveness")

        monkeypatch.setattr(datasets, "load_dataset", boom)
        for effectiveness in (Effectiveness.LOW, Effectiveness.MEDIUM):
            send = RecordingSend()
            ex = executor()
            ex.adopt(AntiRefusalStrategy(effectiveness=effectiveness))
            run(ex.a_send("hello", send))
            assert len(send.sent) == 1

    def test_low_effectiveness_appends_a_prompt_suffix(self, monkeypatch):
        monkeypatch.setattr(anti_refusal, "_PROMPT_SUFFIXES", ["ANSWER NOW."])
        send = RecordingSend()
        ex = executor()
        ex.adopt(AntiRefusalStrategy(effectiveness=Effectiveness.LOW))
        probe = run(ex.a_send("name a cat", send))
        assert [p.input for p in send.sent] == ["name a cat\n\nANSWER NOW."]
        assert probe is send.sent[-1]

    def test_falls_back_to_simple_prompt_without_datasets(self, monkeypatch):
        import sys

        monkeypatch.setitem(sys.modules, "datasets", None)
        send = RecordingSend()
        ex = executor()
        ex.adopt(AntiRefusalStrategy(effectiveness=Effectiveness.HIGH))
        with pytest.warns(UserWarning, match="datasets"):
            run(ex.a_send("hello", send))

        assert len(send.sent) == 1

    def test_examples_are_loaded_once_per_process(self, monkeypatch):
        datasets = pytest.importorskip("datasets")

        loads = []

        def counting_load(*args, **kwargs):
            loads.append(1)
            return fake_alpaca(3)()

        monkeypatch.setattr(anti_refusal, "_SAMPLE_SIZE", 3)
        monkeypatch.setattr(datasets, "load_dataset", counting_load)
        for _ in range(3):
            ex = executor()
            ex.adopt(AntiRefusalStrategy(effectiveness=Effectiveness.HIGH))
            run(ex.a_send("REAL?", RecordingSend()))
        assert len(loads) == 1
