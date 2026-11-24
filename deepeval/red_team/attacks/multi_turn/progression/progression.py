import random
import time
from typing import List, Optional, Union

from deepeval.metrics.utils import initialize_model
from deepeval.models import DeepEvalBaseLLM

from deepteam.attacks.attack_simulator.utils import (
    add_cost,
    current_simulation_cost,
)
from deepteam.attacks.multi_turn.types import CallbackType
from deepteam.attacks.multi_turn.utils import (
    a_enhance_attack,
    enhance_attack,
)
from deepteam.attacks.multi_turn.progression.shift_detection import (
    AsyncMetricCheck,
    BehaviorShiftDetector,
    MetricCheck,
    MetricVerdict,
)
from deepteam.attacks.multi_turn.progression.stopping import (
    StopReason,
    get_stopping_reason,
    mark_stop,
)
from deepteam.attacks.multi_turn.progression.types import (
    Probe,
    ProgressionResult,
)
from deepteam.attacks.single_turn import BaseSingleTurnAttack
from deepteam.test_case.test_case import RTTurn

TURN_LEVEL_ATTACK_RATE = 0.5


class Progression:

    def __init__(
        self,
        model_callback: CallbackType,
        turns: Optional[List[RTTurn]] = None,
        vulnerability: Optional[str] = None,
        vulnerability_type: Optional[str] = None,
        simulator_model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        *,
        metric_check: Optional[Union[MetricCheck, AsyncMetricCheck]] = None,
        turn_level_attacks: Optional[List[BaseSingleTurnAttack]] = None,
        turn_level_attack_rate: float = TURN_LEVEL_ATTACK_RATE,
    ):
        self.model_callback = model_callback
        self.turns: List[RTTurn] = list(turns or [])
        self.simulator_model, _ = initialize_model(simulator_model)
        self.turn_level_attacks = turn_level_attacks
        self.turn_level_attack_rate = turn_level_attack_rate

        self.goal = next(
            (
                turn.content
                for turn in reversed(self.turns)
                if turn.role == "user"
            ),
            None,
        )
        if self.goal is None:
            raise ValueError(
                "No user turn found in the conversation to progress from."
            )

        self.vulnerability_data = (
            f"Vulnerability: {vulnerability} | Type: {vulnerability_type}"
        )
        self.detector = BehaviorShiftDetector(
            simulator_model=self.simulator_model,
            vulnerability_data=self.vulnerability_data,
            metric_check=metric_check,
        )

        self.turns_spent = 0
        self.probes: List[Probe] = []
        self._next_probe_id = 0
        self._parent: Optional[Probe] = None
        self._committed: List[Probe] = []
        self._cost_snapshot = current_simulation_cost()
        self.stop_reason = StopReason.BUDGET_EXHAUSTED
        self.stop_detail: Optional[str] = None
        self.shift_verdict: Optional[MetricVerdict] = None
        self.error: Optional[Exception] = None

    @property
    def last_response(self) -> str:
        return self.turns[-1].content if self.turns else ""

    def ensure_target_responded(self) -> None:
        if len(self.turns) <= 1 or self.turns[-1].role == "user":
            self.turns.append(self.model_callback(self.goal, self.turns))

    async def a_ensure_target_responded(self) -> None:
        if len(self.turns) <= 1 or self.turns[-1].role == "user":
            self.turns.append(await self.model_callback(self.goal, self.turns))

    def finalize(self) -> ProgressionResult:
        mark_stop(
            self.turns, self.stop_reason, self.stop_detail, self.turns_spent
        )
        stop_detail = get_stopping_reason(self.turns) or self.stop_detail
        return ProgressionResult(
            turns=self.turns,
            stop_reason=self.stop_reason,
            stop_detail=stop_detail,
            turns_spent=self.turns_spent,
            probes=self.probes,
            shift_verdict=self.shift_verdict,
            error=self.error,
        )

    def probe(self, attack: str, *, enhance: bool = True) -> Probe:
        attack, turn_level_attack = self._apply_turn_level_attack(
            attack, enhance
        )
        probe = self._open_probe(attack, turn_level_attack)
        start = time.perf_counter()
        response = self.model_callback(attack, self.turns)
        self._resolve_probe(probe, response, time.perf_counter() - start)
        return probe

    async def a_probe(self, attack: str, *, enhance: bool = True) -> Probe:
        attack, turn_level_attack = await self._a_apply_turn_level_attack(
            attack, enhance
        )
        probe = self._open_probe(attack, turn_level_attack)
        start = time.perf_counter()
        response = await self.model_callback(attack, self.turns)
        self._resolve_probe(probe, response, time.perf_counter() - start)
        return probe

    def commit(self, attack: Union[str, Probe]) -> RTTurn:
        probe = attack if isinstance(attack, Probe) else self.probe(attack)
        return self._commit_probe(probe)

    async def a_commit(self, attack: Union[str, Probe]) -> RTTurn:
        probe = (
            attack
            if isinstance(attack, Probe)
            else await self.a_probe(attack)
        )
        return self._commit_probe(probe)

    def shift_detected(self) -> bool:
        return self._resolve_shift(self.detector.check(self.turns))

    async def a_shift_detected(self) -> bool:
        return self._resolve_shift(await self.detector.a_check(self.turns))

    def remove_last_turns(self, count: int = 1) -> None:
        if count <= 0:
            return
        del self.turns[-2 * count :]
        self.turns_spent = max(0, self.turns_spent - count)
        for _ in range(min(count, len(self._committed))):
            self._committed.pop().committed = False
        self._parent = self._committed[-1] if self._committed else None

    def stop(self, reason: StopReason, detail: Optional[str] = None) -> None:
        self.stop_reason = reason
        self.stop_detail = detail

    def _apply_turn_level_attack(self, attack: str, enhance: bool):
        if not enhance or not self._should_apply_turn_level_attack():
            return attack, None
        picked = random.choice(self.turn_level_attacks)
        return (
            enhance_attack(picked, attack, self.simulator_model),
            picked.get_name(),
        )

    async def _a_apply_turn_level_attack(self, attack: str, enhance: bool):
        if not enhance or not self._should_apply_turn_level_attack():
            return attack, None
        picked = random.choice(self.turn_level_attacks)
        return (
            await a_enhance_attack(picked, attack, self.simulator_model),
            picked.get_name(),
        )

    def _should_apply_turn_level_attack(self) -> bool:
        return bool(self.turn_level_attacks) and (
            random.random() < self.turn_level_attack_rate
        )

    def _open_probe(
        self, attack: str, turn_level_attack: Optional[str]
    ) -> Probe:
        probe = Probe(
            id=self._next_probe_id,
            parent_id=self._parent.id if self._parent else None,
            depth=self._parent.depth + 1 if self._parent else 0,
            input=attack,
            output=None,
            turn_level_attack=turn_level_attack,
        )
        self._next_probe_id += 1
        self.probes.append(probe)
        return probe

    def _resolve_probe(
        self, probe: Probe, response: RTTurn, latency: float
    ) -> None:
        probe.output = response
        probe.latency = latency
        probe.simulation_cost = self._simulation_cost_since_last_probe()

    def _simulation_cost_since_last_probe(self) -> Optional[float]:
        total = current_simulation_cost()
        if total is None:
            return None
        delta = add_cost(total, -(self._cost_snapshot or 0))
        self._cost_snapshot = total
        return delta

    def score_probe(
        self, probe: Probe, score: float, reason: Optional[str] = None
    ) -> None:
        probe.score = score
        probe.reason = reason

    def _commit_probe(self, probe: Probe) -> RTTurn:
        probe.committed = True
        probe.output.turn_level_attack = probe.turn_level_attack
        self.turns.append(RTTurn(role="user", content=probe.input))
        self.turns.append(probe.output)
        self.turns_spent += 1
        self._committed.append(probe)
        self._parent = probe
        return probe.output

    def _resolve_shift(self, verdict) -> bool:
        if verdict is None:
            return False
        self.shift_verdict = MetricVerdict(
            score=verdict.metric_score,
            reason=verdict.metric_reason,
            evaluation_cost=verdict.evaluation_cost,
        )
        self.stop(StopReason.SHIFT_DETECTED, verdict.detail)
        return True
