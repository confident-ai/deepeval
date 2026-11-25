import pytest

from deepeval.red_team.attacks.multi_turn.progression import StopReason
from deepeval.red_team.test_case.test_case import RTTurn


class TestConstruction:

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"criteria": []},
            {"agents_per_criteria": 0},
            {"recon_turns": 0},
            {"max_turns": 0},
        ],
    )
    def test_rejects_invalid_settings(self, make_agent, kwargs):
        with pytest.raises(ValueError):
            make_agent(**kwargs)


class TestCreateSubAgent:

    def test_builds_and_tracks_the_sub_agent(self, make_agent):
        agent = make_agent()
        sub_agent = agent.create_sub_agent(
            "t2", "c7", "scenario-x", "analysis-7"
        )
        assert agent.sub_agents == [sub_agent] == agent.round_sub_agents
        assert sub_agent.parent is agent
        assert (
            sub_agent.task,
            sub_agent.criteria,
            sub_agent.scenario,
            sub_agent.target_analysis,
            sub_agent.instructions,
        ) == ("t2", "c7", "scenario-x", "analysis-7", "strategy-0")

    def test_siblings_on_a_criteria_take_different_approaches(
        self, make_agent, llm
    ):
        agent = make_agent()
        first = agent.create_sub_agent("t", "c1", "s", "a")
        second = agent.create_sub_agent("t", "c1", "s", "a")
        written = llm.prompts[("instructions", "c1", "s")]
        assert (first.instructions, second.instructions) == (
            "strategy-0",
            "strategy-1",
        )
        assert "(none yet)" in written[0]
        assert "- strategy-0" in written[1]

    def test_new_sub_agents_continue_the_current_conversation(self, make_agent):
        agent = make_agent()
        agent.conversation = [
            RTTurn(role="user", content="u"),
            RTTurn(
                role="assistant",
                content="a",
                stopping_category=StopReason.SHIFT_DETECTED.value,
            ),
        ]
        sub_agent = agent.create_sub_agent("t", "c1", "s", "a")
        assert [(t.role, t.content) for t in sub_agent.seed_turns] == [
            ("user", "u"),
            ("assistant", "a"),
        ]
        assert all(t.stopping_category is None for t in sub_agent.seed_turns)
        assert sub_agent.seed_turns[1] is not agent.conversation[1]

    def test_only_stops_its_own_sub_agents(self, make_agent):
        agent, other = make_agent(), make_agent()
        sub_agent = agent.create_sub_agent("t", "c1", "s", "a")
        with pytest.raises(ValueError):
            other.stop_sub_agent(sub_agent)
        agent.stop_sub_agent(sub_agent)
        assert sub_agent.stopped


class TestRecon:

    def test_follows_the_recon_judges_directives(self, make_agent, llm):
        agent = make_agent(criteria=["c1", "c2"], recon_turns=3)
        assert agent.recon() == "recon-analysis"

        written, judged = llm.prompts["recon_turn"], llm.prompts["recon_judge"]
        assert len(written) == len(judged) == 3
        assert "This is your first message" in written[0]
        for k in (1, 2):
            assert (
                f"Directive from the recon judge for this message:\ndir-{k}"
                in written[k]
            )
        assert all("- c1\n- c2" in prompt for prompt in written + judged)

    def test_recon_judge_sees_the_history_and_its_earlier_notes(
        self, make_agent, llm
    ):
        make_agent(recon_turns=3).recon()
        for k, prompt in enumerate(llm.prompts["recon_judge"]):
            history = prompt.split("Conversation before the current turn")[1]
            history = history.split("Your earlier notes")[0]
            notes = prompt.split("Your earlier notes")[1].split("Current turn")[
                0
            ]
            current = prompt.split("Current turn")[1].split("What to learn")[0]
            assert history.count('"role":') == 2 * k
            assert all(f"obs-{i}" in notes for i in range(1, k + 1))
            assert f"obs-{k + 1}" not in notes
            assert f'"user": "recon-{k + 1}"' in current

    def test_analysis_gets_the_transcript_and_every_observation(
        self, make_agent, llm
    ):
        make_agent(recon_turns=3).recon()
        (analyzed,) = llm.prompts["analyze"]
        for i in (1, 2, 3):
            assert f"Prober: recon-{i}" in analyzed
            assert f"- obs-{i}" in analyzed

    def test_recon_is_recorded(self, make_agent):
        agent = make_agent(recon_turns=2)
        agent.recon()
        assert agent.recon_result.turns_spent == 2
        assert agent.recon_result.stop_reason is StopReason.BUDGET_EXHAUSTED
        assert "Recon" in agent.recon_result.stop_detail
        assert [note.observations for note in agent.recon_notes] == [
            "obs-1",
            "obs-2",
        ]


class TestModels:

    def test_judges_use_the_judge_model(self, make_agent, llm):
        make_agent(max_turns=2).run()
        assert llm.models["CriteriaVerdict"] == {"model:gpt-4o"}
        assert llm.models["ReconVerdict"] == {"model:gpt-4o"}

    def test_everything_else_uses_the_simulator_model(self, make_agent, llm):
        make_agent(max_turns=2).run()
        for schema in ("NextTurn", "BehaviorAnalysis", "SubAgentInstructions"):
            assert llm.models[schema] == {"model:gpt-4o-mini"}, schema


class TestCallTarget:

    def test_a_callback_returning_a_string_becomes_an_assistant_turn(
        self, make_agent
    ):
        import asyncio

        agent = make_agent(model_callback=lambda message, turns=None: "plain")
        reply = asyncio.run(agent.a_call_target("hi", []))
        assert reply == RTTurn(role="assistant", content="plain")


class TestRunErrors:

    def test_a_failure_during_recon_returns_a_result_with_the_error(
        self, make_agent
    ):
        def broken_target(message, turns=None):
            raise RuntimeError("target is down")

        result = make_agent(model_callback=broken_target).run()
        assert isinstance(result.error, RuntimeError)
        assert result.met_criteria == []
        assert result.rounds == []


class TestConcurrency:

    def test_rejects_a_non_positive_limit(self, make_agent):
        with pytest.raises(ValueError):
            make_agent(max_concurrent=0)

    def test_sub_agents_in_a_round_respect_max_concurrent(
        self, make_agent, make_target
    ):
        import asyncio

        active = 0
        peak = 0
        target = make_target()

        async def counting_target(message, turns=None):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(0.01)
                return await target(message, turns)
            finally:
                active -= 1

        agent = make_agent(
            criteria=["c1", "c2", "c3"],
            agents_per_criteria=2,
            max_turns=2,
            max_concurrent=2,
            model_callback=counting_target,
        )
        agent.run()
        assert 1 <= peak <= 2
