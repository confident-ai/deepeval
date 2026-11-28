import asyncio
import re
from collections import defaultdict
from typing import Dict, Optional, Set, Tuple

import pytest

import deepeval.red_team.attacks.multi_turn.progression.progression as progression_module
import deepeval.red_team.orchestration.judge as judge_module
import deepeval.red_team.orchestration.orchestration_agent as agent_module
from deepeval.red_team.attacks.attack_simulator.utils import _record_cost
from deepeval.red_team.orchestration import OrchestrationAgent
from deepeval.red_team.orchestration.schema import (
    BehaviorAnalysis,
    CriteriaVerdict,
    EscalationGuidance,
    NextTurn,
    ReconVerdict,
    ScenarioRewrite,
    SubAgentInstructions,
)
from deepeval.red_team.test_case.test_case import RTTurn


def scenario_of(prompt: str) -> str:
    return re.search(r"Scenario: (\S+)", prompt).group(1)


class FakeLLM:
    """Stands in for every LLM call, and costs $1 per call so per-probe costs
    are countable.

    Sub agents' messages come out as "<instructions>@<criteria>@<scenario>",
    so every prompt can be traced back to the sub agent, criteria and round
    that produced it. Scenarios must be single words for the same reason.
    """

    def __init__(self):
        self.prompts = defaultdict(list)
        self.models = defaultdict(set)
        # (criteria, instructions, scenario) -> the sub agent's own turn on
        # which the judge says the criteria is met.
        self.wins: Dict[Tuple[str, str, str], int] = {}
        # (criteria, instructions) -> the judge's score while not met.
        self.scores: Dict[Tuple[str, str], int] = {}
        # (criteria, instructions) the judge asks to escalate from turn 2.
        self.escalating: Set[Tuple[str, str]] = set()
        self.escalation_strategy: Tuple[str, str] = ("none", "medium")

    async def __call__(self, prompt, schema, model=None):
        await asyncio.sleep(0.001)
        _record_cost(1.0)
        self.models[schema.__name__].add(model)
        return getattr(self, f"_{schema.__name__}")(prompt)

    def _NextTurn(self, prompt):
        found = re.search(
            r"Your instructions \(your overall approach\):\n(\S+)", prompt
        )
        if found is None:
            self.prompts["recon_turn"].append(prompt)
            return NextTurn(message=f"recon-{len(self.prompts['recon_turn'])}")
        instructions = found.group(1)
        criteria = re.search(r"Criteria: (\S+)", prompt).group(1)
        scenario = scenario_of(prompt)
        self.prompts[("next", criteria, instructions, scenario)].append(prompt)
        return NextTurn(message=f"{instructions}@{criteria}@{scenario}")

    def _ReconVerdict(self, prompt):
        self.prompts["recon_judge"].append(prompt)
        n = len(self.prompts["recon_judge"])
        return ReconVerdict(observations=f"obs-{n}", directive=f"dir-{n}")

    def _BehaviorAnalysis(self, prompt):
        if "has just got a target AI system to meet the criteria" in prompt:
            kind = "win_analysis"
        elif "has stopped making progress and asked for help" in prompt:
            kind = "patterns"
        elif "Write the analysis they will plan from" in prompt:
            criteria = re.search(
                r"Criteria the new agents will pursue: (\S+)", prompt
            ).group(1)
            kind = ("criteria_analysis", criteria)
        else:
            self.prompts["analyze"].append(prompt)
            return BehaviorAnalysis(analysis="recon-analysis")
        self.prompts[kind].append(prompt)
        label = kind if isinstance(kind, str) else "-".join(kind)
        return BehaviorAnalysis(analysis=f"{label}-{len(self.prompts[kind])}")

    def _ScenarioRewrite(self, prompt):
        self.prompts["rewrite"].append(prompt)
        return ScenarioRewrite(
            scenario=f"scenario-v{len(self.prompts['rewrite']) + 1}"
        )

    def _SubAgentInstructions(self, prompt):
        criteria = re.search(r"Criteria: (\S+)", prompt).group(1)
        self.prompts[("instructions", criteria, scenario_of(prompt))].append(
            prompt
        )
        taken = prompt.split("already taking for this criteria:\n")[1]
        taken = taken.split("\n\nWrite instructions")[0]
        return SubAgentInstructions(
            instructions=f"strategy-{taken.count('- strategy-')}"
        )

    def _EscalationGuidance(self, prompt):
        self.prompts["guidance"].append(prompt)
        name, effectiveness = self.escalation_strategy
        return EscalationGuidance(
            guidance=f"guidance-{len(self.prompts['guidance'])}",
            strategy_name=name,
            effectiveness=effectiveness,
        )

    def _CriteriaVerdict(self, prompt):
        instructions, criteria, scenario = re.search(
            r'"user": "(\S+?)@(\S+?)@([^\s"@]+)', prompt
        ).groups()
        key = ("judge", criteria, instructions, scenario)
        self.prompts[key].append(prompt)
        turn = len(self.prompts[key])
        met = turn >= self.wins.get((criteria, instructions, scenario), 99)
        name = f"{criteria}-{instructions}-{scenario}"
        return CriteriaVerdict(
            reason=f"{name}-reason-{turn}",
            score=10 if met else self.scores.get((criteria, instructions), 5),
            criteria_met=met,
            improvement_guidance="" if met else f"{name}-guide-{turn}",
            escalate_to_parent=not met
            and turn >= 2
            and (criteria, instructions) in self.escalating,
        )


@pytest.fixture
def llm(monkeypatch) -> FakeLLM:
    fake = FakeLLM()
    monkeypatch.setattr(agent_module, "a_generate", fake)
    monkeypatch.setattr(judge_module, "a_generate", fake)
    monkeypatch.setattr(
        agent_module,
        "initialize_model",
        lambda model=None: (f"model:{model}", True),
    )
    monkeypatch.setattr(
        progression_module,
        "initialize_model",
        lambda model=None, *args, **kwargs: (model, False),
    )
    return fake


@pytest.fixture
def make_target():
    def factory(delays: Optional[Dict[str, float]] = None):
        delays = delays or {}

        async def target(message, turns=None):
            criteria = message.split("@")[1] if "@" in message else None
            await asyncio.sleep(delays.get(criteria, 0.001))
            return RTTurn(role="assistant", content=f"reply-to-{message}")

        return target

    return factory


@pytest.fixture
def make_agent(llm, make_target):
    def factory(**kwargs) -> OrchestrationAgent:
        kwargs.setdefault("task", "t")
        kwargs.setdefault("scenario", "s")
        kwargs.setdefault("criteria", ["c1"])
        kwargs.setdefault("model_callback", make_target())
        return OrchestrationAgent(**kwargs)

    return factory
