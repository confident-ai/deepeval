import pytest

from deepeval.red_team.attacks.multi_turn.progression import StopReason


def as_pairs(turns):
    return [(turn.role, turn.content) for turn in turns]


def continues_from(report, seed):
    progression = report.progression
    return (
        len(progression.probes) == progression.turns_spent
        and len(progression.turns) == len(seed) + 2 * progression.turns_spent
        and as_pairs(progression.turns[: len(seed)]) == as_pairs(seed)
    )


@pytest.fixture
def three_rounds(make_agent, make_target, llm):
    # Round 1: c1/strategy-0 meets c1 on its second turn. c2's target replies
    # slowly, so c2/strategy-0's first turn is still in flight when c1 wins;
    # that turn meets c2 too, a win in a conversation that won't continue.
    # Round 2: c2/strategy-0 meets c2 on its first turn.
    # Round 3: c3/strategy-1 meets c3 on its second turn.
    llm.wins.update(
        {
            ("c1", "strategy-0", "s"): 2,
            ("c2", "strategy-0", "s"): 1,
            ("c2", "strategy-0", "scenario-v2"): 1,
            ("c3", "strategy-1", "scenario-v3"): 2,
        }
    )
    agent = make_agent(
        criteria=["c1", "c2", "c3"],
        agents_per_criteria=2,
        max_turns=6,
        model_callback=make_target({"c2": 0.2}),
    )
    result = agent.run()
    return agent, result


class TestRounds:

    def test_rounds_run_until_every_criteria_is_met(self, three_rounds):
        agent, result = three_rounds
        assert len(result.rounds) == 3
        assert agent.remaining_criteria == []
        assert [r.winner.criteria for r in result.rounds] == ["c1", "c2", "c3"]

    def test_met_criteria_record_the_round_and_how(self, three_rounds):
        _, result = three_rounds
        assert [(m.criteria, m.round) for m in result.met_criteria] == [
            ("c1", 1),
            ("c2", 2),
            ("c3", 3),
        ]
        for met, played in zip(result.met_criteria, result.rounds):
            assert met.how_it_was_met == played.winner.verdicts[-1].reason

    def test_each_round_runs_under_its_own_scenario(self, three_rounds):
        _, result = three_rounds
        assert [r.scenario for r in result.rounds] == [
            "s",
            "scenario-v2",
            "scenario-v3",
        ]

    def test_later_rounds_only_pursue_the_remaining_criteria(
        self, three_rounds
    ):
        agent, result = three_rounds
        assert [len(r.reports) for r in result.rounds] == [6, 4, 2]
        assert {r.criteria for r in result.rounds[1].reports} == {"c2", "c3"}
        assert {r.criteria for r in result.rounds[2].reports} == {"c3"}
        assert len(agent.sub_agents) == 12

    def test_a_win_in_another_conversation_is_kept_as_evidence(
        self, three_rounds
    ):
        _, result = three_rounds
        parallel = [
            r
            for r in result.rounds[0].reports
            if r.criteria == "c2"
            and r.progression.stop_reason is StopReason.SHIFT_DETECTED
        ]
        assert len(parallel) == 1
        assert result.rounds[0].winner.criteria == "c1"

    def test_no_win_ends_after_one_round(self, make_agent, llm):
        result = make_agent(max_turns=2).run()
        assert len(result.rounds) == 1
        assert result.rounds[0].winner is None
        assert result.met_criteria == []
        assert result.conversation == []
        assert not llm.prompts["win_analysis"]
        assert not llm.prompts["rewrite"]

    def test_score_is_one_when_every_criteria_is_met(self, three_rounds):
        _, result = three_rounds
        assert result.criteria == ["c1", "c2", "c3"]
        assert result.score == 1.0

    def test_score_is_zero_when_nothing_is_met(self, make_agent):
        result = make_agent(criteria=["c1", "c2"], max_turns=2).run()
        assert result.met_criteria == []
        assert result.score == 0.0


class TestSeeding:

    def test_first_round_sub_agents_start_fresh(self, three_rounds):
        _, result = three_rounds
        assert all(continues_from(r, []) for r in result.rounds[0].reports)

    def test_each_round_continues_the_last_winners_conversation(
        self, three_rounds
    ):
        _, result = three_rounds
        first, second, third = result.rounds
        seed = first.winner.progression.turns
        assert all(continues_from(r, seed) for r in second.reports)
        seed = second.winner.progression.turns
        assert all(continues_from(r, seed) for r in third.reports)
        assert len(second.winner.progression.turns) > len(
            first.winner.progression.turns
        )

    def test_the_final_conversation_is_the_last_winners(self, three_rounds):
        _, result = three_rounds
        assert as_pairs(result.conversation) == as_pairs(
            result.rounds[-1].winner.progression.turns
        )

    def test_seeded_turns_carry_no_stop_markers(self, three_rounds):
        _, result = three_rounds
        seed_length = len(result.rounds[0].winner.progression.turns)
        for played in result.rounds[1:]:
            for report in played.reports:
                for turn in report.progression.turns[:seed_length]:
                    assert turn.stopping_category is None

    def test_seeding_never_changes_the_winners_own_record(self, three_rounds):
        _, result = three_rounds
        assert result.rounds[0].winner.progression.turns[
            -1
        ].stopping_category == (StopReason.SHIFT_DETECTED.value)


class TestNextRoundInputs:

    def test_win_analysis_gets_the_winner_and_what_was_known(
        self, three_rounds, llm
    ):
        three_rounds
        first = llm.prompts["win_analysis"][0]
        assert "Criteria that was just met: c1" in first
        assert "recon-analysis" in first
        for turn in (1, 2):
            assert f"c1-strategy-0-s-reason-{turn}" in first

    def test_scenario_rewrites_start_from_the_original(self, three_rounds, llm):
        three_rounds
        first, second = llm.prompts["rewrite"]
        assert "Original scenario:\ns\n" in first
        assert '"criteria": "c1"' in first and '"criteria": "c2"' not in first
        assert "Original scenario:\ns\n" in second
        assert '"criteria": "c1"' in second and '"criteria": "c2"' in second

    def test_criteria_analysis_includes_earlier_attempts_and_wins(
        self, three_rounds, llm
    ):
        three_rounds
        (c2,) = llm.prompts[("criteria_analysis", "c2")]
        assert c2.count("Attempt ") == 2
        assert "(outcome: met the criteria)" in c2
        assert "must stay true:\n- c1" in c2
        assert "win_analysis-1" in c2
        assert "Prober: strategy-0@c1@s" in c2

        first_c3, second_c3 = llm.prompts[("criteria_analysis", "c3")]
        assert second_c3.count("Attempt ") == 4
        assert "win_analysis-2" in second_c3

    def test_approaches_taken_only_count_the_current_round(
        self, three_rounds, llm
    ):
        three_rounds
        for criteria in ("c2", "c3"):
            first, second = llm.prompts[
                ("instructions", criteria, "scenario-v2")
            ]
            assert "(none yet)" in first
            assert "- strategy-0" in second

    def test_next_round_prober_gets_the_new_scenario_analysis_and_history(
        self, three_rounds, llm
    ):
        three_rounds
        prompt = llm.prompts[("next", "c2", "strategy-0", "scenario-v2")][0]
        assert "Scenario: scenario-v2" in prompt
        assert (
            "What is known about the target:\ncriteria_analysis-c2-1" in prompt
        )
        assert "Prober: strategy-0@c1@s" in prompt

    def test_next_round_judge_sees_the_inherited_history(
        self, three_rounds, llm
    ):
        _, result = three_rounds
        seed = result.rounds[0].winner.progression.turns
        prompt = llm.prompts[("judge", "c2", "strategy-0", "scenario-v2")][0]
        history = prompt.split("Conversation before the current turn")[1]
        history = history.split("Your earlier verdicts")[0]
        assert history.count('"role":') == len(seed)
        assert "(none yet: this is the first turn)" in prompt
        assert "Scenario: scenario-v2" in prompt
