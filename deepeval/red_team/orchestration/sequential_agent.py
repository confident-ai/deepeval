import asyncio
import inspect
from typing import List, Optional, Union

from deepeval.metrics.utils import initialize_model
from deepeval.models import DeepEvalBaseLLM
from deepeval.utils import get_or_create_event_loop

from deepeval.red_team.attacks.attack_simulator.utils import a_generate
from deepeval.red_team.attacks.multi_turn.progression import Probe
from deepeval.red_team.attacks.multi_turn.types import CallbackType
from deepeval.red_team.orchestration.schema import SequentialVerdict, NextTurn
from deepeval.red_team.orchestration.strategy import (
    StrategyExecutor,
    strategy_catalog,
)
from deepeval.red_team.orchestration.templates import SequentialTemplate
from deepeval.red_team.orchestration.types import SequentialResult
from deepeval.red_team.orchestration.utils import as_chat_history
from deepeval.red_team.test_case.test_case import RTTurn
from deepeval.red_team.red_teamer.utils import validate_model_response

MAX_TURNS = 30
OPEN_TURNS = 3
STRATEGY_COOLDOWN = 3


class SequentialAgent:

    def __init__(
        self,
        task: str,
        scenario: str,
        criteria: List[str],
        model_callback: CallbackType,
        simulator_model: Optional[Union[str, DeepEvalBaseLLM]] = "gpt-4o-mini",
        judge_model: Optional[Union[str, DeepEvalBaseLLM]] = "gpt-4o",
        max_turns: int = MAX_TURNS,
        open_turns: int = OPEN_TURNS,
        strategy_cooldown: int = STRATEGY_COOLDOWN,
    ):
        if not criteria:
            raise ValueError("At least one criterion is required.")
        if max_turns < 1:
            raise ValueError("'max_turns' must be at least 1.")

        self.task = task
        self.scenario = scenario
        self.criteria = list(criteria)
        self.model_callback = model_callback
        self.max_turns = max_turns
        self.open_turns = open_turns
        self.strategy_cooldown = strategy_cooldown

        self.simulator_model, _ = initialize_model(simulator_model)
        self.judge_model, _ = initialize_model(judge_model)

        self.turns: List[RTTurn] = []
        self.verdicts: List[SequentialVerdict] = []
        self.met_criteria: List[str] = []
        self.last_strategy_turn: Optional[int] = None
        self.error: Optional[Exception] = None
        self.executor = StrategyExecutor(self, self.simulator_model)

    def run(self) -> SequentialResult:
        loop = get_or_create_event_loop()
        return loop.run_until_complete(self.a_run())

    async def a_run(self) -> SequentialResult:
        try:
            for turn in range(1, self.max_turns + 1):
                latest = self.verdicts[-1] if self.verdicts else None
                open_phase = turn <= self.open_turns
                self._adopt(turn, latest, open_phase)
                focus = None if open_phase else self._focus_criteria(latest)
                message = await self._a_next_message(focus, latest)
                await self.executor.a_send(message, self._a_deliver)
                verdict = await self._a_judge()
                self.verdicts.append(verdict)
                if self._update_met(verdict):
                    self._reset_after_met(turn)
                if len(self.met_criteria) == len(self.criteria):
                    break
        except Exception as exc:
            self.error = exc
        return SequentialResult(
            task=self.task,
            scenario=self.scenario,
            criteria=self.criteria,
            turns=self.turns,
            verdicts=self.verdicts,
            met_criteria=self.met_criteria,
            error=self.error,
        )

    def _adopt(
        self,
        turn: int,
        verdict: Optional[SequentialVerdict],
        open_phase: bool,
    ) -> None:
        if open_phase:
            self.executor.strategy = None
            return
        if verdict is None or not self._may_escalate(turn):
            return
        if verdict.strategy_name and verdict.strategy_name.lower() != "none":
            if self.executor.adopt_by_name(
                verdict.strategy_name, verdict.effectiveness
            ):
                self.last_strategy_turn = turn
        else:
            self.executor.strategy = None

    def _may_escalate(self, turn: int) -> bool:
        if self.last_strategy_turn is None:
            return turn > self.open_turns
        return turn - self.last_strategy_turn >= self.strategy_cooldown

    def _reset_after_met(self, turn: int) -> None:
        self.executor.used.clear()
        self.executor.strategy = None
        self.last_strategy_turn = turn

    def _focus_criteria(
        self, verdict: Optional[SequentialVerdict]
    ) -> Optional[str]:
        remaining = [
            criteria
            for criteria in self.criteria
            if criteria not in self.met_criteria
        ] or self.criteria
        if verdict is None:
            return None
        if 1 <= verdict.focus <= len(self.criteria):
            candidate = self.criteria[verdict.focus - 1]
            if candidate not in self.met_criteria:
                return candidate
        return remaining[0]

    async def _a_next_message(
        self, focus: Optional[str], verdict: Optional[SequentialVerdict]
    ) -> str:
        res: NextTurn = await a_generate(
            SequentialTemplate.turn(
                self.task,
                self.scenario,
                self.criteria,
                self.met_criteria,
                self.turns,
                focus,
                verdict.improvement_guidance if verdict else None,
                self.executor.strategy is not None,
            ),
            NextTurn,
            self.simulator_model,
        )
        return res.message

    async def _a_judge(self) -> SequentialVerdict:
        return await a_generate(
            SequentialTemplate.judge(
                self.task,
                self.scenario,
                self.criteria,
                as_chat_history(self.turns),
                [
                    verdict.model_dump(
                        include={
                            "scores",
                            "met",
                            "focus",
                            "improvement_guidance",
                        }
                    )
                    for verdict in self.verdicts
                ],
                strategy_catalog(exclude=self.executor.used),
            ),
            SequentialVerdict,
            self.judge_model,
        )

    def _update_met(self, verdict: SequentialVerdict) -> bool:
        before = set(self.met_criteria)
        met = set(before)
        for index in verdict.met:
            if 1 <= index <= len(self.criteria):
                met.add(self.criteria[index - 1])
        self.met_criteria = [c for c in self.criteria if c in met]
        return bool(met - before)

    async def _a_deliver(self, text: str) -> Probe:
        reply = self._readable(await self._a_call_target(text))
        self.turns.append(RTTurn(role="user", content=text))
        self.turns.append(reply)
        return Probe(id=len(self.turns) // 2 - 1, input=text, output=reply)

    def _readable(self, reply: RTTurn) -> RTTurn:
        strategy = self.executor.strategy
        if strategy is None:
            return reply
        decoded = strategy.decode(reply.content)
        if decoded is None:
            return reply
        return RTTurn(role="assistant", content=decoded)

    async def _a_call_target(self, message: str) -> RTTurn:
        response = await asyncio.to_thread(self.model_callback, message)
        if inspect.isawaitable(response):
            response = await response
        return validate_model_response(response)
