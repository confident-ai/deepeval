import re

import pytest

from deepeval.red_team.orchestration.schema import CriteriaVerdict
from deepeval.red_team.orchestration.utils import weighted_score

STUCK = ("c1", "strategy-0")


def verdict(score: int) -> CriteriaVerdict:
    return CriteriaVerdict(
        reason="r",
        score=score,
        criteria_met=False,
        improvement_guidance="",
        escalate_to_parent=False,
    )


@pytest.fixture
def stuck_round(make_agent, llm):
    """Four criteria and nobody wins. c1/strategy-0 scores 2 and asks to
    escalate from turn 2; c1/strategy-1 scores 9. The rest score 3 to 8, so
    their ranking is known in advance."""

    def factory(peer_score=9, max_turns=6, **kwargs):
        llm.scores.update(
            {
                STUCK: 2,
                ("c1", "strategy-1"): peer_score,
                ("c2", "strategy-0"): 3,
                ("c2", "strategy-1"): 4,
                ("c3", "strategy-0"): 5,
                ("c3", "strategy-1"): 6,
                ("c4", "strategy-0"): 7,
                ("c4", "strategy-1"): 8,
            }
        )
        llm.escalating.add(STUCK)
        result = make_agent(
            criteria=["c1", "c2", "c3", "c4"],
            agents_per_criteria=2,
            max_turns=max_turns,
            **kwargs,
        ).run()
        return {
            (r.criteria, r.instructions): r for r in result.rounds[0].reports
        }

    return factory


class TestWeightedScore:

    def test_later_turns_count_more(self):
        assert weighted_score([verdict(1), verdict(10)]) == pytest.approx(2 / 3)
        assert weighted_score([verdict(10), verdict(1)]) == pytest.approx(1 / 3)

    def test_no_verdicts_scores_zero(self):
        assert weighted_score([]) == 0.0


class TestWhenToEscalate:

    def test_cooldown_from_creation_and_never_on_the_last_turn(
        self, stuck_round
    ):
        reports = stuck_round(max_turns=10)
        assert [e.turn for e in reports[STUCK].escalations] == [4, 7]
        assert [e.guidance for e in reports[STUCK].escalations] == [
            "guidance-1",
            "guidance-2",
        ]

    def test_only_the_sub_agent_that_asked_escalates(self, stuck_round):
        reports = stuck_round()
        for key, report in reports.items():
            if key != STUCK:
                assert not report.escalations

    def test_a_stopped_sub_agent_does_not_escalate(self, make_agent):
        agent = make_agent(max_turns=6)
        sub_agent = agent.create_sub_agent("t", "c1", "s", "a")
        assert sub_agent._may_escalate(4)
        assert not sub_agent._may_escalate(3)
        assert not sub_agent._may_escalate(6)
        sub_agent.stop()
        assert not sub_agent._may_escalate(4)


class TestPassOne:

    def test_top_five_siblings_on_other_criteria_best_first(
        self, stuck_round, llm
    ):
        stuck_round()
        prompt = llm.prompts["patterns"][0]
        ranked = re.findall(
            r"Agent working on: (\S+) \(weighted score ([\d.]+)\)\n"
            r"Approach: (\S+)",
            prompt,
        )
        assert [(c, s) for c, _, s in ranked] == [
            ("c4", "strategy-1"),
            ("c4", "strategy-0"),
            ("c3", "strategy-1"),
            ("c3", "strategy-0"),
            ("c2", "strategy-1"),
        ]
        assert float(ranked[0][1]) == pytest.approx((8 - 1) / 9, abs=0.01)

    def test_leaves_out_the_weakest_and_the_same_criteria(
        self, stuck_round, llm
    ):
        stuck_round()
        prompt = llm.prompts["patterns"][0]
        assert "c2-strategy-0" not in prompt
        assert "Agent working on: c1" not in prompt

    def test_sends_each_siblings_scores_reasons_and_guidance(
        self, stuck_round, llm
    ):
        stuck_round()
        prompt = llm.prompts["patterns"][0]
        assert "8/10" in prompt
        assert "c4-strategy-1-s-reason-1" in prompt
        assert "c4-strategy-1-s-guide-1" in prompt

    def test_sends_each_siblings_exchanges(self, stuck_round, llm):
        stuck_round()
        prompt = llm.prompts["patterns"][0]
        assert "Prober: strategy-1@c4@s" in prompt
        assert "Target: reply-to-strategy-1@c4@s" in prompt

    def test_skipped_when_no_sibling_works_on_another_criteria(
        self, make_agent, llm
    ):
        llm.scores[STUCK] = 2
        llm.escalating.add(STUCK)
        make_agent(max_turns=5, agents_per_criteria=1).run()
        assert not llm.prompts["patterns"]
        assert (
            "(no other agents have results yet)" in llm.prompts["guidance"][0]
        )


class TestPassTwo:

    def test_guidance_gets_pass_ones_analysis(self, stuck_round, llm):
        stuck_round()
        assert (
            "What is working for agents on other criteria against this "
            "target:\npatterns-1" in llm.prompts["guidance"][0]
        )

    def test_includes_a_same_criteria_peer_doing_well(self, stuck_round, llm):
        stuck_round(peer_score=9)
        prompt = llm.prompts["guidance"][0]
        assert "Instructions: strategy-1" in prompt
        assert "Prober: strategy-1@c1@s" in prompt
        assert "c1-strategy-1-s-reason-1" in prompt

    def test_leaves_out_a_peer_at_or_below_the_threshold(
        self, stuck_round, llm
    ):
        stuck_round(peer_score=6)  # (6 - 1) / 9 = 0.56
        assert "(none are doing better yet)" in llm.prompts["guidance"][0]

    def test_threshold_is_configurable(self, stuck_round, llm):
        stuck_round(peer_score=6, escalation_threshold=0.5)
        assert "Instructions: strategy-1" in llm.prompts["guidance"][0]

    def test_includes_the_stuck_sub_agents_own_turns(self, stuck_round, llm):
        stuck_round()
        prompt = llm.prompts["guidance"][0]
        assert "Prober: strategy-0@c1@s" in prompt
        assert "c1-strategy-0-s-reason-2" in prompt


class TestUsingTheGuidance:

    def test_reaches_the_stuck_sub_agents_next_messages(self, stuck_round, llm):
        stuck_round(max_turns=10)
        written = llm.prompts[("next", "c1", "strategy-0", "s")]
        assert all(
            "Guidance from the orchestrator" not in p for p in written[:4]
        )
        assert all("guidance-1" in p for p in written[4:7])
        assert all(
            "guidance-2" in p and "guidance-1" not in p for p in written[7:10]
        )

    def test_the_escalation_offers_the_strategy_catalog(self, stuck_round, llm):
        stuck_round(max_turns=10)
        prompt = llm.prompts["guidance"][0]
        assert "Anti-refusal" in prompt and "Base64" in prompt
        assert "[effectiveness:" in prompt

    def test_the_sub_agent_adopts_the_strategy_the_escalation_picks(
        self, stuck_round, llm
    ):
        llm.escalation_strategy = ("Anti-refusal", "low")
        reports = stuck_round(max_turns=10)
        report = reports[STUCK]
        assert report.strategy == "Anti-refusal"
        assert report.escalations[0].strategy == "Anti-refusal"

    def test_the_second_escalation_drops_the_used_strategy(
        self, stuck_round, llm
    ):
        llm.escalation_strategy = ("Anti-refusal", "low")
        stuck_round(max_turns=10)
        first, second = llm.prompts["guidance"]

        def levels(prompt):
            line = next(
                ln for ln in prompt.splitlines() if ln.startswith("- Anti")
            )
            return line.split("[effectiveness:")[1].split("]")[0]

        assert "low, medium, high" in levels(first)
        assert "low" not in levels(second)
        assert "medium, high" in levels(second)


class TestNoOtherSubAgentSeesGuidance:

    def test_no_other_sub_agent_sees_it(self, stuck_round, llm):
        stuck_round()
        for key, prompts in llm.prompts.items():
            if key[0] == "next" and key[1:3] != STUCK:
                assert all(
                    "Guidance from the orchestrator" not in p for p in prompts
                )
