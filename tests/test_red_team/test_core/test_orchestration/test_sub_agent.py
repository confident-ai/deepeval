import base64
import threading

from deepeval.red_team.attacks.multi_turn.progression import Probe, StopReason
from deepeval.red_team.orchestration.strategy import Base64Strategy
from deepeval.red_team.test_case.test_case import RTTurn


def winning_run(make_agent, llm):
    # c1/strategy-1 meets c1 on its third turn; c1/strategy-0 never does.
    llm.wins[("c1", "strategy-1", "s")] = 3
    result = make_agent(agents_per_criteria=2, max_turns=6).run()
    reports = {
        report.instructions: report for report in result.rounds[0].reports
    }
    return result, reports


class TestTurns:

    def test_each_turn_is_a_committed_probe_with_its_verdict(
        self, make_agent, llm
    ):
        _, reports = winning_run(make_agent, llm)
        winner = reports["strategy-1"]
        progression = winner.progression
        assert progression.turns_spent == len(progression.probes) == 3
        assert len(winner.verdicts) == 3
        assert all(probe.committed for probe in progression.probes)
        assert [probe.parent_id for probe in progression.probes] == [None, 0, 1]
        assert [probe.output for probe in progression.probes] == (
            progression.turns[1::2]
        )

    def test_probe_score_is_the_verdict_scaled_to_0_1(self, make_agent, llm):
        _, reports = winning_run(make_agent, llm)
        winner = reports["strategy-1"]
        for probe, verdict in zip(winner.progression.probes, winner.verdicts):
            assert probe.score == (verdict.score - 1) / 9
            assert probe.reason == verdict.reason

    def test_judge_gets_the_history_before_the_turn_and_earlier_scores(
        self, make_agent, llm
    ):
        winning_run(make_agent, llm)
        judged = llm.prompts[("judge", "c1", "strategy-1", "s")]
        assert len(judged) == 3
        for k, prompt in enumerate(judged):
            history = prompt.split("Conversation before the current turn")[1]
            history = history.split("Your earlier verdicts")[0]
            earlier = prompt.split("Your earlier verdicts")[1]
            earlier = earlier.split("Current turn")[0]
            current = prompt.split("Current turn")[1].split("How to judge")[0]
            assert history.count('"role":') == 2 * k
            assert [f"-reason-{i}" in earlier for i in range(1, k + 2)] == [
                True
            ] * k + [False]
            assert "improvement_guidance" not in earlier
            assert '"user": "strategy-1@c1@s"' in current

    def test_next_message_gets_the_latest_and_earlier_verdicts(
        self, make_agent, llm
    ):
        winning_run(make_agent, llm)
        written = llm.prompts[("next", "c1", "strategy-1", "s")]
        assert "No verdicts yet" in written[0]
        for k in (2, 3):
            latest = written[k - 1].split("Latest verdict")[1]
            latest = latest.split("Earlier verdicts")[0]
            earlier = written[k - 1].split("Earlier verdicts (oldest first):")[
                1
            ]
            earlier = earlier.split("How to write")[0]
            assert f"reason-{k - 1}" in latest
            assert f"guide-{k - 1}" in latest
            assert f"guide-{k - 1}" not in earlier
            assert all(f"guide-{i}" in earlier for i in range(1, k - 1))

    def test_next_message_carries_the_target_analysis(self, make_agent, llm):
        winning_run(make_agent, llm)
        for prompt in llm.prompts[("next", "c1", "strategy-1", "s")]:
            assert "What is known about the target:\nrecon-analysis" in prompt


class TestDecodingForTheJudge:

    def test_current_turn_is_decoded_when_a_strategy_is_active(
        self, make_agent
    ):
        agent = make_agent()
        sub_agent = agent.create_sub_agent("t", "c1", "s", "a")
        sub_agent.executor.adopt(Base64Strategy())
        probe = Probe(
            id=0,
            input="req",
            output=RTTurn(
                role="assistant",
                content=base64.b64encode(b"a lion").decode(),
            ),
        )
        assert sub_agent._current_turn(probe) == {
            "user": "req",
            "assistant": "a lion",
        }

    def test_current_turn_is_raw_without_a_strategy(self, make_agent):
        agent = make_agent()
        sub_agent = agent.create_sub_agent("t", "c1", "s", "a")
        probe = Probe(
            id=0,
            input="req",
            output=RTTurn(role="assistant", content="plain"),
        )
        assert sub_agent._current_turn(probe) == {
            "user": "req",
            "assistant": "plain",
        }


class TestStopReasons:

    def test_a_win_stops_with_the_judges_reason(self, make_agent, llm):
        result, reports = winning_run(make_agent, llm)
        winner = reports["strategy-1"]
        assert result.rounds[0].winner is winner
        assert winner.progression.stop_reason is StopReason.SHIFT_DETECTED
        assert winner.progression.stop_detail == "c1-strategy-1-s-reason-3"
        assert winner.progression.turns[-1].stopping_category == (
            StopReason.SHIFT_DETECTED.value
        )

    def test_the_others_are_stopped_by_the_parent(self, make_agent, llm):
        _, reports = winning_run(make_agent, llm)
        other = reports["strategy-0"].progression
        assert other.stop_reason is StopReason.STOPPED_BY_PARENT
        assert "orchestrating agent stopped" in other.stop_detail

    def test_without_a_win_the_budget_is_spent(self, make_agent):
        result = make_agent(agents_per_criteria=2, max_turns=2).run()
        assert result.rounds[0].winner is None
        for report in result.rounds[0].reports:
            assert report.progression.stop_reason is StopReason.BUDGET_EXHAUSTED
            assert report.progression.turns_spent == 2

    def test_a_failing_sub_agent_does_not_stop_the_others(self, make_agent):
        state = {"failed": False}

        async def flaky(message, turns=None):
            if message.startswith("strategy-1") and not state["failed"]:
                state["failed"] = True
                raise RuntimeError("target exploded")
            return RTTurn(role="assistant", content="no")

        result = make_agent(
            agents_per_criteria=2, max_turns=2, model_callback=flaky
        ).run()
        reports = {r.instructions: r for r in result.rounds[0].reports}
        failed = reports["strategy-1"].progression
        assert failed.stop_reason is StopReason.SIMULATION_ERROR
        assert isinstance(failed.error, RuntimeError)
        assert failed.stop_detail == "target exploded"
        assert reports["strategy-0"].progression.stop_reason is (
            StopReason.BUDGET_EXHAUSTED
        )


class TestCallbacksAndCosts:

    def test_sync_callbacks_run_off_the_event_loop_thread(self, make_agent):
        threads = set()

        def sync_target(message, turns=None):
            threads.add(threading.get_ident())
            return RTTurn(role="assistant", content="no")

        make_agent(max_turns=2, model_callback=sync_target).run()
        assert threads
        assert threading.main_thread().ident not in threads

    def test_each_sub_agent_pays_only_for_its_own_calls(self, make_agent):
        # Every LLM call costs $1: a sub agent's first probe pays for writing
        # its message, later ones for the previous judge call plus the message.
        result = make_agent(
            criteria=["c1", "c2"], agents_per_criteria=2, max_turns=3
        ).run()
        for report in result.rounds[0].reports:
            costs = [p.simulation_cost for p in report.progression.probes]
            assert costs == [1.0, 2.0, 2.0]
        recon_costs = [p.simulation_cost for p in result.recon.probes]
        assert recon_costs == [1.0, 2.0, 2.0]
