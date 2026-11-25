from typing import TYPE_CHECKING, List, Optional, Tuple

from deepeval.red_team.attacks.multi_turn.progression import Probe
from deepeval.red_team.orchestration.strategy.strategy import Send, Strategy

if TYPE_CHECKING:
    from deepeval.red_team.orchestration.sub_agent import SubAgent


class StrategyExecutor:

    def __init__(self, sub_agent: "SubAgent", simulator_model=None):
        self.sub_agent = sub_agent
        self.simulator_model = simulator_model
        self.strategy: Optional[Strategy] = None
        self.used: List[Tuple[str, str]] = []

    def adopt(self, strategy: Strategy) -> None:
        strategy.generate = self._a_generate
        self.strategy = strategy
        self.used.append((strategy.name, strategy.effectiveness.value))

    async def _a_generate(self, prompt: str) -> str:
        from deepeval.red_team.attacks.attack_simulator.utils import a_generate
        from deepeval.red_team.orchestration.schema import NextTurn

        res = await a_generate(prompt, NextTurn, self.simulator_model)
        return res.message

    def adopt_by_name(
        self, name: Optional[str], effectiveness: Optional[str]
    ) -> Optional[Strategy]:
        from deepeval.red_team.orchestration.strategy.registry import (
            build_strategy,
        )

        strategy = build_strategy(name, effectiveness)
        if strategy is not None:
            self.adopt(strategy)
        return strategy

    def has_used(self, name: str, effectiveness: str) -> bool:
        return (name, effectiveness) in self.used

    async def a_send(self, message: str, send: Send) -> Probe:
        if self.strategy is None:
            return await send(message)

        delivered: List[Probe] = []

        async def tracked(text: str) -> Probe:
            probe = await send(text)
            delivered.append(probe)
            return probe

        await self.strategy.a_execute(message, tracked)

        if not delivered:
            return await send(message)
        return delivered[-1]
