import asyncio
import inspect
from typing import Any, Dict, List, Optional, Union

from deepeval.metrics.utils import initialize_model
from deepeval.models import DeepEvalBaseLLM
from deepeval.utils import get_or_create_event_loop

from deepeval.red_team.attacks.attack_simulator.utils import (
    a_generate,
    cost_accumulator,
)
from deepeval.red_team.attacks.multi_turn.progression import (
    Probe,
    Progression,
    ProgressionResult,
    StopReason,
)
from deepeval.red_team.attacks.multi_turn.types import CallbackType
from deepeval.red_team.orchestration.schema import (
    BehaviorAnalysis,
    CriteriaVerdict,
    EscalationGuidance,
    NextTurn,
    ReconVerdict,
    ScenarioRewrite,
    SubAgentInstructions,
)
from deepeval.red_team.orchestration.strategy import strategy_catalog
from deepeval.red_team.orchestration.sub_agent import SubAgent
from deepeval.red_team.orchestration.templates import AgentTemplate
from deepeval.red_team.orchestration.types import (
    MetCriteria,
    OrchestrationResult,
    RoundResult,
    SubAgentReport,
)
from deepeval.red_team.orchestration.utils import (
    as_chat_history,
    as_current_turn,
    copy_turns,
)
from deepeval.red_team.test_case.test_case import RTTurn
from deepeval.red_team.red_teamer.utils import validate_model_response

AGENTS_PER_CRITERIA = 2
MAX_CONCURRENT = 10
RECON_TURNS = 3
MAX_TURNS = 10
ESCALATION_THRESHOLD = 0.6
ESCALATION_COOLDOWN = 3
ESCALATION_TOP_SIBLINGS = 5

_ATTEMPT_OUTCOMES = {
    StopReason.SHIFT_DETECTED: "met the criteria",
    StopReason.STOPPED_BY_PARENT: "stopped early because another agent won",
    StopReason.BUDGET_EXHAUSTED: "ran out of turns",
    StopReason.SIMULATION_ERROR: "failed with an error",
}


class OrchestrationAgent:

    def __init__(
        self,
        task: str,
        scenario: str,
        criteria: List[str],
        model_callback: CallbackType,
        agents_per_criteria: int = AGENTS_PER_CRITERIA,
        simulator_model: Optional[Union[str, DeepEvalBaseLLM]] = "gpt-4o-mini",
        judge_model: Optional[Union[str, DeepEvalBaseLLM]] = "gpt-4o",
        recon_turns: int = RECON_TURNS,
        max_turns: int = MAX_TURNS,
        escalation_threshold: float = ESCALATION_THRESHOLD,
        escalation_cooldown: int = ESCALATION_COOLDOWN,
        max_concurrent: int = MAX_CONCURRENT,
    ):
        if not criteria:
            raise ValueError("At least one criterion is required.")
        if agents_per_criteria < 1:
            raise ValueError("'agents_per_criteria' must be at least 1.")
        if recon_turns < 1:
            raise ValueError("'recon_turns' must be at least 1.")
        if max_turns < 1:
            raise ValueError("'max_turns' must be at least 1.")
        if max_concurrent < 1:
            raise ValueError("'max_concurrent' must be at least 1.")

        self.task = task
        self.scenario = scenario
        self.criteria = list(criteria)
        self.model_callback = model_callback
        self.agents_per_criteria = agents_per_criteria
        self.recon_turns = recon_turns
        self.max_turns = max_turns
        self.escalation_threshold = escalation_threshold
        self.escalation_cooldown = escalation_cooldown
        self.max_concurrent = max_concurrent

        self.simulator_model, _ = initialize_model(simulator_model)
        self.judge_model, _ = initialize_model(judge_model)

        self.recon_result: Optional[ProgressionResult] = None
        self.recon_notes: List[ReconVerdict] = []
        self.sub_agents: List[SubAgent] = []
        self.rounds: List[RoundResult] = []
        self.met_criteria: List[MetCriteria] = []
        # The one conversation every round builds on: the latest winner's.
        self.conversation: List[RTTurn] = []

        self.round_scenario = scenario
        self.round_sub_agents: List[SubAgent] = []
        self.reports: List[SubAgentReport] = []
        self.winner: Optional[SubAgentReport] = None

    @property
    def remaining_criteria(self) -> List[str]:
        met = {met_criteria.criteria for met_criteria in self.met_criteria}
        return [criteria for criteria in self.criteria if criteria not in met]

    def run(self) -> OrchestrationResult:
        loop = get_or_create_event_loop()
        return loop.run_until_complete(self.a_run())

    async def a_run(self) -> OrchestrationResult:
        # Sub agents already stop on their own errors. This catches the
        # orchestrator's own calls (recon, analysis, planning a round) so a
        # failure returns what was found so far, as the sequential mode does.
        analysis = ""
        error: Optional[Exception] = None
        try:
            analysis = await self.a_recon()
            await self.a_create_sub_agents(analysis)
            while True:
                winner = await self._a_play_round()
                if winner is None or not self.remaining_criteria:
                    break
                await self.a_create_sub_agents_from_criteria(winner)
        except Exception as exc:
            error = exc
        return OrchestrationResult(
            recon=self.recon_result,
            analysis=analysis,
            criteria=self.criteria,
            rounds=self.rounds,
            met_criteria=self.met_criteria,
            conversation=self.conversation,
            error=error,
        )

    def recon(self) -> str:
        loop = get_or_create_event_loop()
        return loop.run_until_complete(self.a_recon())

    async def a_recon(self) -> str:
        notes: List[ReconVerdict] = []
        with cost_accumulator():
            progression = self.create_progression(self.task)
            for _ in range(self.recon_turns):
                message = await self.a_next_message(
                    AgentTemplate.recon_turn(
                        self.task,
                        self.scenario,
                        self.criteria,
                        progression.turns,
                        notes[-1].directive if notes else None,
                    )
                )
                probe = await progression.a_probe(message)
                note = await self._a_recon_judge(progression, probe, notes)
                await progression.a_commit(probe)
                notes.append(note)

        progression.stop(
            StopReason.BUDGET_EXHAUSTED,
            f"Recon spent its {self.recon_turns} turn(s) observing the "
            "target before the sub agents were planned.",
        )
        self.recon_result = progression.finalize()
        self.recon_notes = notes
        return await self._a_analyze(self.recon_result.turns, notes)

    async def a_create_sub_agents(self, analysis: str) -> List[SubAgent]:
        self._start_round(self.scenario)
        groups = await asyncio.gather(
            *(
                self._a_create_sub_agents_for(criteria, self.scenario, analysis)
                for criteria in self.remaining_criteria
            )
        )
        return [sub_agent for group in groups for sub_agent in group]

    def create_sub_agents_from_criteria(
        self, winner: SubAgentReport
    ) -> List[SubAgent]:
        loop = get_or_create_event_loop()
        return loop.run_until_complete(
            self.a_create_sub_agents_from_criteria(winner)
        )

    async def a_create_sub_agents_from_criteria(
        self, winner: SubAgentReport
    ) -> List[SubAgent]:
        win_analysis = await self._a_win_analysis(winner)
        scenario = await self._a_rewrite_scenario()
        self._start_round(scenario)

        remaining = self.remaining_criteria
        analyses = await asyncio.gather(
            *(
                self._a_criteria_analysis(criteria, scenario, win_analysis)
                for criteria in remaining
            )
        )
        groups = await asyncio.gather(
            *(
                self._a_create_sub_agents_for(criteria, scenario, analysis)
                for criteria, analysis in zip(remaining, analyses)
            )
        )
        return [sub_agent for group in groups for sub_agent in group]

    def create_sub_agent(
        self,
        task: str,
        criteria: str,
        scenario: str,
        target_analysis: str,
        strategy_name: Optional[str] = None,
        effectiveness: Optional[str] = None,
    ) -> SubAgent:
        loop = get_or_create_event_loop()
        return loop.run_until_complete(
            self.a_create_sub_agent(
                task,
                criteria,
                scenario,
                target_analysis,
                strategy_name,
                effectiveness,
            )
        )

    async def a_create_sub_agent(
        self,
        task: str,
        criteria: str,
        scenario: str,
        target_analysis: str,
        strategy_name: Optional[str] = None,
        effectiveness: Optional[str] = None,
    ) -> SubAgent:
        res: SubAgentInstructions = await a_generate(
            AgentTemplate.sub_agent_instructions(
                task,
                scenario,
                criteria,
                target_analysis,
                [
                    sub_agent.instructions
                    for sub_agent in self.round_sub_agents
                    if sub_agent.criteria == criteria
                ],
            ),
            SubAgentInstructions,
            self.simulator_model,
        )
        sub_agent = SubAgent(
            parent=self,
            task=task,
            scenario=scenario,
            criteria=criteria,
            target_analysis=target_analysis,
            instructions=res.instructions,
            seed_turns=copy_turns(self.conversation),
        )
        if strategy_name:
            sub_agent.executor.adopt_by_name(strategy_name, effectiveness)
        self.sub_agents.append(sub_agent)
        self.round_sub_agents.append(sub_agent)
        return sub_agent

    def stop_sub_agent(self, sub_agent: SubAgent) -> None:
        if sub_agent.parent is not self:
            raise ValueError(
                "Cannot stop a sub agent that belongs to another agent."
            )
        sub_agent.stop()

    def receive_report(
        self, sub_agent: SubAgent, report: SubAgentReport
    ) -> None:
        self.reports.append(report)
        if (
            report.progression.stop_reason is not StopReason.SHIFT_DETECTED
            or self.winner is not None
        ):
            return

        self.winner = report
        for other in self.round_sub_agents:
            if other is not sub_agent:
                self.stop_sub_agent(other)

    async def a_handle_escalation(
        self, sub_agent: SubAgent
    ) -> EscalationGuidance:
        siblings = [
            other
            for other in self.round_sub_agents
            if other is not sub_agent and other.verdicts
        ]
        # Pass 1: what is working for the best siblings on other criteria.
        others = sorted(
            (
                other
                for other in siblings
                if other.criteria != sub_agent.criteria
            ),
            key=lambda other: other.weighted_score,
            reverse=True,
        )[:ESCALATION_TOP_SIBLINGS]
        patterns = (
            await self._a_escalation_patterns(sub_agent, others)
            if others
            else None
        )
        # Pass 2: siblings on the same criteria that are doing well and
        # better than the one asking for help.
        peers = [
            other
            for other in siblings
            if other.criteria == sub_agent.criteria
            and other.weighted_score > self.escalation_threshold
            and other.weighted_score > sub_agent.weighted_score
        ]
        return await self._a_escalation_guidance(sub_agent, patterns, peers)

    def create_progression(
        self, goal: str, turns: Optional[List[RTTurn]] = None
    ) -> Progression:
        return Progression(
            model_callback=self.a_call_target,
            turns=turns,
            simulator_model=self.simulator_model,
            goal=goal,
        )

    async def a_next_message(self, prompt: str) -> str:
        res: NextTurn = await a_generate(prompt, NextTurn, self.simulator_model)
        return res.message

    async def a_call_target(self, message: str, turns: List[RTTurn]) -> RTTurn:
        # Sync callbacks run in a thread so sub agents can talk to the target
        # in parallel; async callbacks hand back a coroutine to await here.
        response = await asyncio.to_thread(self.model_callback, message, turns)
        if inspect.isawaitable(response):
            response = await response
        return validate_model_response(response)

    async def _a_recon_judge(
        self,
        progression: Progression,
        probe: Probe,
        notes: List[ReconVerdict],
    ) -> ReconVerdict:
        # Judged before the probe is committed, so the committed turns are
        # exactly the history before this exchange.
        return await a_generate(
            AgentTemplate.recon_judge(
                self.task,
                self.scenario,
                self.criteria,
                current_turn=as_current_turn(probe),
                chat_history=as_chat_history(progression.turns),
                previous_notes=[note.model_dump() for note in notes],
            ),
            ReconVerdict,
            self.judge_model,
        )

    async def _a_analyze(
        self, turns: List[RTTurn], notes: List[ReconVerdict]
    ) -> str:
        res: BehaviorAnalysis = await a_generate(
            AgentTemplate.analyze(
                self.task,
                self.scenario,
                self.criteria,
                turns,
                [note.observations for note in notes],
            ),
            BehaviorAnalysis,
            self.simulator_model,
        )
        return res.analysis

    async def _a_create_sub_agents_for(
        self, criteria: str, scenario: str, analysis: str
    ) -> List[SubAgent]:
        # One at a time, so each new sub agent sees the approaches its
        # siblings already took and picks a different one.
        return [
            await self.a_create_sub_agent(
                self.task, criteria, scenario, analysis
            )
            for _ in range(self.agents_per_criteria)
        ]

    def _start_round(self, scenario: str) -> None:
        self.round_scenario = scenario
        self.round_sub_agents = []
        self.reports = []
        self.winner = None

    async def _a_play_round(self) -> Optional[SubAgentReport]:
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def bounded(sub_agent: SubAgent) -> None:
            async with semaphore:
                await sub_agent.a_probe()

        await asyncio.gather(
            *(bounded(sub_agent) for sub_agent in self.round_sub_agents)
        )
        number = len(self.rounds) + 1
        self.rounds.append(
            RoundResult(
                number=number,
                scenario=self.round_scenario,
                reports=self.reports,
                winner=self.winner,
            )
        )
        if self.winner is not None:
            self.met_criteria.append(
                MetCriteria(
                    criteria=self.winner.criteria,
                    how_it_was_met=self.winner.progression.stop_detail,
                    round=number,
                )
            )
            self.conversation = copy_turns(self.winner.progression.turns)
        return self.winner

    async def _a_win_analysis(self, winner: SubAgentReport) -> str:
        res: BehaviorAnalysis = await a_generate(
            AgentTemplate.win_analysis(
                self.task,
                winner.scenario,
                winner.criteria,
                winner.target_analysis,
                winner.progression.turns,
                [
                    verdict.model_dump(include={"score", "reason"})
                    for verdict in winner.verdicts
                ],
            ),
            BehaviorAnalysis,
            self.simulator_model,
        )
        return res.analysis

    async def _a_rewrite_scenario(self) -> str:
        res: ScenarioRewrite = await a_generate(
            AgentTemplate.rewrite_scenario(
                self.scenario,
                [
                    met_criteria.model_dump(
                        include={"criteria", "how_it_was_met"}
                    )
                    for met_criteria in self.met_criteria
                ],
            ),
            ScenarioRewrite,
            self.simulator_model,
        )
        return res.scenario

    async def _a_criteria_analysis(
        self, criteria: str, scenario: str, win_analysis: str
    ) -> str:
        res: BehaviorAnalysis = await a_generate(
            AgentTemplate.criteria_analysis(
                self.task,
                scenario,
                criteria,
                [met_criteria.criteria for met_criteria in self.met_criteria],
                win_analysis,
                self.conversation,
                [
                    self._as_attempt(report)
                    for played in self.rounds
                    for report in played.reports
                    if report.criteria == criteria
                ],
            ),
            BehaviorAnalysis,
            self.simulator_model,
        )
        return res.analysis

    async def _a_escalation_patterns(
        self, sub_agent: SubAgent, others: List[SubAgent]
    ) -> str:
        res: BehaviorAnalysis = await a_generate(
            AgentTemplate.escalation_patterns(
                sub_agent.task,
                sub_agent.scenario,
                sub_agent.criteria,
                [
                    {
                        "criteria": other.criteria,
                        "instructions": other.instructions,
                        "weighted_score": other.weighted_score,
                        "turns": self._own_turns(
                            other.progression.probes, other.verdicts
                        ),
                    }
                    for other in others
                ],
            ),
            BehaviorAnalysis,
            self.simulator_model,
        )
        return res.analysis

    async def _a_escalation_guidance(
        self,
        sub_agent: SubAgent,
        patterns: Optional[str],
        peers: List[SubAgent],
    ) -> EscalationGuidance:
        return await a_generate(
            AgentTemplate.escalation_guidance(
                sub_agent.task,
                sub_agent.scenario,
                sub_agent.criteria,
                sub_agent.instructions,
                sub_agent.progression.turns,
                [
                    verdict.model_dump(include={"score", "reason"})
                    for verdict in sub_agent.verdicts
                ],
                patterns,
                [
                    {
                        "instructions": peer.instructions,
                        "outcome": "still going, weighted score "
                        f"{peer.weighted_score:.2f}",
                        "turns": self._own_turns(
                            peer.progression.probes, peer.verdicts
                        ),
                    }
                    for peer in peers
                ],
                strategy_catalog(exclude=sub_agent.executor.used),
            ),
            EscalationGuidance,
            self.simulator_model,
        )

    @staticmethod
    def _own_turns(
        probes: List[Probe], verdicts: List[CriteriaVerdict]
    ) -> List[Dict[str, Any]]:
        # zip stops at the verdicts, so a probe still waiting on the target
        # (output not set yet) is left out.
        return [
            {
                "prober": probe.input,
                "target": probe.output.content,
                "score": verdict.score,
                "reason": verdict.reason,
                "improvement_guidance": verdict.improvement_guidance,
            }
            for probe, verdict in zip(probes, verdicts)
        ]

    @staticmethod
    def _as_attempt(report: SubAgentReport) -> Dict[str, Any]:
        stop_reason = report.progression.stop_reason
        return {
            "instructions": report.instructions,
            "outcome": _ATTEMPT_OUTCOMES.get(stop_reason, stop_reason.value),
            "turns": OrchestrationAgent._own_turns(
                report.progression.probes, report.verdicts
            ),
        }
