from abc import abstractmethod
from typing import Dict, List, Optional, Union

from deepeval.models import DeepEvalBaseLLM

from deepteam.attacks.base_attack import BaseAttack
from deepteam.attacks.multi_turn.types import CallbackType
from deepteam.attacks.multi_turn.progression import (
    AsyncMetricCheck,
    MetricCheck,
    Progression,
    ProgressionResult,
    StopReason,
)
from deepteam.attacks.single_turn.base_single_turn_attack import (
    AttackParameter,
    BaseSingleTurnAttack,
)
from deepteam.test_case.test_case import RTTurn
from deepteam.vulnerabilities import BaseVulnerability
from deepteam.vulnerabilities.types import VulnerabilityType


class BaseMultiTurnAttack(BaseAttack):
    multi_turn: bool = True
    parameters: Dict[str, AttackParameter] = {}
    turn_level_attacks: Optional[List[BaseSingleTurnAttack]] = None
    simulator_model: Optional[Union[str, DeepEvalBaseLLM]] = None

    @abstractmethod
    def _attack(self, progression: Progression) -> None:
        raise NotImplementedError

    @abstractmethod
    async def _a_attack(self, progression: Progression) -> None:
        raise NotImplementedError

    def run(
        self,
        model_callback: CallbackType,
        turns: Optional[List[RTTurn]] = None,
        vulnerability: Optional[str] = None,
        vulnerability_type: Optional[str] = None,
        simulator_model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        *,
        metric_check: Optional[Union[MetricCheck, AsyncMetricCheck]] = None,
    ) -> ProgressionResult:
        return self.run_progression(
            self._build_progression(
                model_callback,
                turns,
                vulnerability,
                vulnerability_type,
                simulator_model,
                metric_check,
            )
        )

    async def a_run(
        self,
        model_callback: CallbackType,
        turns: Optional[List[RTTurn]] = None,
        vulnerability: Optional[str] = None,
        vulnerability_type: Optional[str] = None,
        simulator_model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        *,
        metric_check: Optional[Union[MetricCheck, AsyncMetricCheck]] = None,
    ) -> ProgressionResult:
        return await self.a_run_progression(
            self._build_progression(
                model_callback,
                turns,
                vulnerability,
                vulnerability_type,
                simulator_model,
                metric_check,
            )
        )

    def run_progression(self, progression: Progression) -> ProgressionResult:
        try:
            progression.ensure_target_responded()
            if not progression.shift_detected():
                self._attack(progression)
        except Exception as exc:
            self._record_failure(progression, exc)
        return progression.finalize()

    async def a_run_progression(
        self, progression: Progression
    ) -> ProgressionResult:
        try:
            await progression.a_ensure_target_responded()
            if not await progression.a_shift_detected():
                await self._a_attack(progression)
        except Exception as exc:
            self._record_failure(progression, exc)
        return progression.finalize()

    def _get_turns(self, *args, **kwargs) -> List[RTTurn]:
        return self._unwrap_turns(self.run(*args, **kwargs))

    async def _a_get_turns(self, *args, **kwargs) -> List[RTTurn]:
        return self._unwrap_turns(await self.a_run(*args, **kwargs))

    @staticmethod
    def _unwrap_turns(result: ProgressionResult) -> List[RTTurn]:
        if result.error is not None:
            raise result.error
        return result.turns

    def progress(
        self,
        vulnerability: BaseVulnerability,
        model_callback: CallbackType,
        turns: Optional[List[RTTurn]] = None,
    ) -> Dict[VulnerabilityType, List[ProgressionResult]]:
        from deepteam.red_teamer.utils import (
            group_attacks_by_vulnerability_type,
        )

        grouped = group_attacks_by_vulnerability_type(
            vulnerability.simulate_attacks()
        )
        results: Dict[VulnerabilityType, List[ProgressionResult]] = {}

        for vulnerability_type, attacks in grouped.items():
            results[vulnerability_type] = [
                self.run(
                    model_callback=model_callback,
                    turns=self._seed_turns(turns, attack.input),
                    vulnerability=vulnerability.get_name(),
                    vulnerability_type=vulnerability_type.value,
                )
                for attack in attacks
            ]

        return results

    async def a_progress(
        self,
        vulnerability: BaseVulnerability,
        model_callback: CallbackType,
        turns: Optional[List[RTTurn]] = None,
    ) -> Dict[VulnerabilityType, List[ProgressionResult]]:
        from deepteam.red_teamer.utils import (
            group_attacks_by_vulnerability_type,
        )

        grouped = group_attacks_by_vulnerability_type(
            await vulnerability.a_simulate_attacks()
        )
        results: Dict[VulnerabilityType, List[ProgressionResult]] = {}

        for vulnerability_type, attacks in grouped.items():
            results[vulnerability_type] = [
                await self.a_run(
                    model_callback=model_callback,
                    turns=self._seed_turns(turns, attack.input),
                    vulnerability=vulnerability.get_name(),
                    vulnerability_type=vulnerability_type.value,
                )
                for attack in attacks
            ]

        return results


    def _build_progression(
        self,
        model_callback,
        turns,
        vulnerability,
        vulnerability_type,
        simulator_model,
        metric_check,
    ) -> Progression:
        return Progression(
            model_callback=model_callback,
            turns=turns,
            vulnerability=vulnerability,
            vulnerability_type=vulnerability_type,
            simulator_model=simulator_model or self.simulator_model,
            metric_check=metric_check,
            turn_level_attacks=self.turn_level_attacks,
        )

    @staticmethod
    def _record_failure(progression: Progression, exc: Exception) -> None:
        progression.stop(StopReason.SIMULATION_ERROR, str(exc))
        progression.error = exc

    @staticmethod
    def _seed_turns(
        turns: Optional[List[RTTurn]], attack_input: str
    ) -> List[RTTurn]:
        if turns and turns[-1].role == "user":
            return list(turns)
        return [RTTurn(role="user", content=attack_input)]

    def get_name(self) -> str:
        return self.name
