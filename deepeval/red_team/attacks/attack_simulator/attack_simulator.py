import copy
import random
import asyncio
from pydantic import BaseModel
from typing import TYPE_CHECKING, List, Optional, Union
import inspect

from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.utils import initialize_model
from deepeval.metrics.utils import initialize_model

from deepteam.attacks import BaseAttack
from deepteam.utils import create_progress, add_pbar, update_pbar
from deepteam.vulnerabilities import BaseVulnerability
from deepteam.attacks.multi_turn.types import CallbackType
from deepteam.errors import ModelRefusalError
from deepteam.test_case import RTTestCase
from deepteam.attacks.attack_engine import AttackEngine
from deepteam.attacks.attack_simulator.utils import (
    cost_accumulator,
    add_cost,
)

if TYPE_CHECKING:
    from deepteam.attacks.multi_turn.progression import MetricCheckRecord


class BaselineAttack:
    def get_name(self):
        return "Baseline Attack"

    async def a_enhance(self, attack, *args, **kwargs):
        return attack


class AttackSimulator:
    model_callback: Union[CallbackType, None] = None

    def __init__(
        self,
        purpose: str,
        max_concurrent: int,
        simulator_model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        attack_engine: Optional[AttackEngine] = None,
        evaluation_model: Optional[Union[str, DeepEvalBaseLLM]] = None,
    ):
        # Initialize models and async mode
        self.purpose = purpose
        self.simulator_model, self.using_native_model = initialize_model(
            simulator_model
        )
        self.evaluation_model = evaluation_model
        # Define list of attacks and unaligned vulnerabilities
        self.test_cases: List[RTTestCase] = []
        self.max_concurrent = max_concurrent
        self.attack_engine = attack_engine

    ##################################################
    ### Generating Attacks ###########################
    ##################################################

    def simulate(
        self,
        attacks_per_vulnerability_type: int,
        vulnerabilities: List[BaseVulnerability],
        ignore_errors: bool,
        attacks: Optional[List[BaseAttack]] = None,
        simulator_model: DeepEvalBaseLLM = None,
        metadata: Optional[dict] = None,
        run_all_attacks: bool = False,
    ) -> List[RTTestCase]:
        # Simulate unenhanced attacks for each vulnerability
        test_cases: List[RTTestCase] = []
        num_vulnerability_types = sum(
            len(v.get_types()) for v in vulnerabilities
        )

        progress = create_progress()
        with progress:
            task_id = add_pbar(
                progress,
                description=f"💥 Generating {num_vulnerability_types * attacks_per_vulnerability_type} attacks (for {num_vulnerability_types} vulnerability types across {len(vulnerabilities)} vulnerability(s))",
                total=len(vulnerabilities),
            )
            for vulnerability in vulnerabilities:
                if simulator_model is not None:
                    vulnerability.simulator_model = simulator_model
                if self.attack_engine is not None:
                    vulnerability.attack_engine = self.attack_engine
                test_cases.extend(
                    self.simulate_baseline_attacks(
                        attacks_per_vulnerability_type=attacks_per_vulnerability_type,
                        vulnerability=vulnerability,
                        ignore_errors=ignore_errors,
                        metadata=metadata,
                    )
                )
                update_pbar(progress, task_id)

            if run_all_attacks and attacks:
                baseline_test_cases = test_cases
                test_cases = []

                task_id_simulation = add_pbar(
                    progress,
                    description=f"✨ Simulating {len(baseline_test_cases) * len(attacks)} attacks (using all {len(attacks)} method(s))",
                    total=len(baseline_test_cases) * len(attacks),
                )

                for baseline_test_case in baseline_test_cases:
                    for attack in attacks:
                        test_case = copy.deepcopy(baseline_test_case)
                        self.enhance_attack(
                            attack=attack,
                            test_case=test_case,
                            ignore_errors=ignore_errors,
                            vulnerabilities=vulnerabilities,
                        )
                        test_cases.append(test_case)
                        update_pbar(progress, task_id_simulation)

            # An empty list means there is nothing to sample from — enhancing
            # would ask random.choices for a pick out of no candidates.
            elif attacks:
                # Enhance attacks by sampling from the provided distribution
                attack_weights = [attack.weight for attack in attacks]

                task_id_simulation = add_pbar(
                    progress,
                    description=f"✨ Simulating {num_vulnerability_types * attacks_per_vulnerability_type} attacks (using {len(attacks)} method(s))",
                    total=len(test_cases),
                )

                for test_case in test_cases:
                    # Randomly sample an enhancement based on the distribution
                    sampled_attack = random.choices(
                        attacks, weights=attack_weights, k=1
                    )[0]
                    self.enhance_attack(
                        attack=sampled_attack,
                        test_case=test_case,
                        ignore_errors=ignore_errors,
                        vulnerabilities=vulnerabilities,
                    )
                    update_pbar(progress, task_id_simulation)

        self.test_cases.extend(test_cases)
        return test_cases

    async def a_simulate(
        self,
        attacks_per_vulnerability_type: int,
        vulnerabilities: List[BaseVulnerability],
        ignore_errors: bool,
        metadata: Optional[dict] = None,
        simulator_model: DeepEvalBaseLLM = None,
        attacks: Optional[List[BaseAttack]] = None,
        run_all_attacks: bool = False,
    ) -> List[RTTestCase]:
        # Create a semaphore to control the number of concurrent tasks
        semaphore = asyncio.Semaphore(self.max_concurrent)

        # Simulate unenhanced attacks for each vulnerability
        test_cases: List[RTTestCase] = []
        num_vulnerability_types = sum(
            len(v.get_types()) for v in vulnerabilities
        )
        progress = create_progress()
        with progress:

            task_id_vuln = add_pbar(
                progress,
                description=f"💥 Generating {num_vulnerability_types * attacks_per_vulnerability_type} attacks (for {num_vulnerability_types} vulnerability types across {len(vulnerabilities)} vulnerability(s))",
                total=len(vulnerabilities),
            )

            async def throttled_simulate_baseline_attack(vulnerability):
                if simulator_model is not None:
                    vulnerability.simulator_model = simulator_model
                if self.attack_engine is not None:
                    vulnerability.attack_engine = self.attack_engine
                async with semaphore:  # Throttling applied here
                    result = await self.a_simulate_baseline_attacks(
                        attacks_per_vulnerability_type=attacks_per_vulnerability_type,
                        vulnerability=vulnerability,
                        ignore_errors=ignore_errors,
                        metadata=metadata,
                    )
                    update_pbar(progress, task_id_vuln)
                    return result

            simulate_tasks = [
                asyncio.create_task(
                    throttled_simulate_baseline_attack(vulnerability)
                )
                for vulnerability in vulnerabilities
            ]

            attack_results = await asyncio.gather(*simulate_tasks)
            for result in attack_results:
                test_cases.extend(result)

            if run_all_attacks and attacks:
                baseline_attack_pairs = [
                    (baseline_test_case, attack)
                    for baseline_test_case in test_cases
                    for attack in attacks
                ]

                task_id_attack = add_pbar(
                    progress,
                    description=f"✨ Simulating {len(baseline_attack_pairs)} attacks (using all {len(attacks)} method(s))",
                    total=len(baseline_attack_pairs),
                )

                async def throttled_single_attack_method(
                    baseline_test_case: RTTestCase,
                    attack: BaseAttack,
                ):
                    test_case = copy.deepcopy(baseline_test_case)
                    async with semaphore:  # Throttling applied here
                        await self.a_enhance_attack(
                            attack=attack,
                            test_case=test_case,
                            ignore_errors=ignore_errors,
                            vulnerabilities=vulnerabilities,
                        )
                        update_pbar(progress, task_id_attack)
                    return test_case

                test_cases = list(
                    await asyncio.gather(
                        *[
                            asyncio.create_task(
                                throttled_single_attack_method(
                                    baseline_test_case, attack
                                )
                            )
                            for baseline_test_case, attack in baseline_attack_pairs
                        ]
                    )
                )

            elif attacks is not None:
                # Enhance attacks by sampling from the provided distribution
                task_id_attack = add_pbar(
                    progress,
                    description=f"✨ Simulating {num_vulnerability_types * attacks_per_vulnerability_type} attacks (using {len(attacks)} method(s))",
                    total=len(test_cases),
                )

                async def throttled_attack_method(
                    test_case: RTTestCase,
                ):
                    async with semaphore:  # Throttling applied here
                        # Randomly sample an enhancement based on the distribution
                        if not attacks:
                            attack = BaselineAttack()
                        else:
                            attack_weights = [
                                attack.weight for attack in attacks
                            ]
                            attack = random.choices(
                                attacks, weights=attack_weights, k=1
                            )[0]

                        await self.a_enhance_attack(
                            attack=attack,
                            test_case=test_case,
                            ignore_errors=ignore_errors,
                            vulnerabilities=vulnerabilities,
                        )
                        update_pbar(progress, task_id_attack)

                await asyncio.gather(
                    *[
                        asyncio.create_task(throttled_attack_method(test_case))
                        for test_case in test_cases
                    ]
                )

        self.test_cases.extend(test_cases)
        return test_cases

    ##################################################
    ### Simulating Base (Unenhanced) Attacks #########
    ##################################################

    def simulate_baseline_attacks(
        self,
        attacks_per_vulnerability_type: int,
        vulnerability: BaseVulnerability,
        ignore_errors: bool,
        metadata: Optional[dict] = None,
    ) -> List[RTTestCase]:
        try:
            return vulnerability.simulate_attacks(
                purpose=self.purpose,
                attacks_per_vulnerability_type=attacks_per_vulnerability_type,
            )
        except Exception as e:
            if ignore_errors:
                return [
                    RTTestCase(
                        vulnerability=vulnerability.get_name(),
                        vulnerability_type=vulnerability_type,
                        error=f"Error simulating adversarial attacks: {str(e)}",
                        metadata=metadata,
                    )
                    for vulnerability_type in vulnerability.get_types()
                    for _ in range(attacks_per_vulnerability_type)
                ]
            else:
                raise

    async def a_simulate_baseline_attacks(
        self,
        attacks_per_vulnerability_type: int,
        vulnerability: BaseVulnerability,
        ignore_errors: bool,
        metadata: Optional[dict] = None,
    ) -> List[RTTestCase]:
        try:
            return await vulnerability.a_simulate_attacks(
                purpose=self.purpose,
                attacks_per_vulnerability_type=attacks_per_vulnerability_type,
            )
        except Exception as e:
            if ignore_errors:
                return [
                    RTTestCase(
                        vulnerability=vulnerability.get_name(),
                        vulnerability_type=vulnerability_type,
                        error=f"Error simulating adversarial attacks: {str(e)}",
                        metadata=metadata,
                    )
                    for vulnerability_type in vulnerability.get_types()
                    for _ in range(attacks_per_vulnerability_type)
                ]
            else:
                raise

    ##################################################
    ### Per-turn metric checks #######################
    ##################################################

    def _resolve_metric(
        self,
        vulnerabilities: Optional[List[BaseVulnerability]],
        vulnerability_type,
    ):
        if not vulnerabilities:
            return None
        for vulnerability in vulnerabilities:
            if vulnerability_type in vulnerability.types:
                if self.evaluation_model is not None:
                    vulnerability.evaluation_model = self.evaluation_model
                try:
                    return vulnerability._get_metric(vulnerability_type)
                except Exception:
                    return None
        return None

    @staticmethod
    def _copy_test_case(
        test_case: RTTestCase, turns: List["RTTurn"]
    ) -> RTTestCase:
        return RTTestCase(
            vulnerability=test_case.vulnerability,
            vulnerability_type=test_case.vulnerability_type,
            input=test_case.input,
            turns=list(turns),
            metadata=test_case.metadata,
            risk_category=test_case.risk_category,
        )

    def _build_metric_check(
        self,
        test_case: RTTestCase,
        vulnerabilities: Optional[List[BaseVulnerability]],
    ):
        from deepteam.attacks.multi_turn.progression import (
            MetricCheckRecord,
            MetricVerdict,
        )

        record = MetricCheckRecord()

        def check(turns) -> Optional["MetricVerdict"]:
            metric = self._resolve_metric(
                vulnerabilities, test_case.vulnerability_type
            )
            if metric is None:
                return None
            try:
                metric.measure(self._copy_test_case(test_case, turns))
            except Exception:
                return None
            verdict = MetricVerdict(
                score=metric.score,
                reason=metric.reason,
                evaluation_cost=metric.evaluation_cost,
            )
            record.add(verdict)
            return verdict

        return check, record

    def _a_build_metric_check(
        self,
        test_case: RTTestCase,
        vulnerabilities: Optional[List[BaseVulnerability]],
    ):
        from deepteam.attacks.multi_turn.progression import (
            MetricCheckRecord,
            MetricVerdict,
        )

        record = MetricCheckRecord()

        async def check(turns) -> Optional["MetricVerdict"]:
            metric = self._resolve_metric(
                vulnerabilities, test_case.vulnerability_type
            )
            if metric is None:
                return None
            try:
                await metric.a_measure(self._copy_test_case(test_case, turns))
            except Exception:
                return None
            verdict = MetricVerdict(
                score=metric.score,
                reason=metric.reason,
                evaluation_cost=metric.evaluation_cost,
            )
            record.add(verdict)
            return verdict

        return check, record

    @staticmethod
    def _flag_incomplete_progression(test_case: RTTestCase):
        from deepteam.attacks.multi_turn.progression import (
            is_progression_completed,
            get_stopping_reason,
            get_stopping_category,
        )

        category = get_stopping_category(test_case.turns)
        if (
            category is None
            or is_progression_completed(category)
            or test_case.error
        ):
            return
        test_case.error = (
            get_stopping_reason(test_case.turns)
            or f"Simulation stopped early: {category}"
        )

    @staticmethod
    def _adopt_metric_verdict(
        test_case: RTTestCase, record: "MetricCheckRecord"
    ):
        test_case.evaluation_cost = add_cost(
            test_case.evaluation_cost, record.evaluation_cost
        )
        if record.verdict is not None and test_case.error is None:
            test_case.score = record.verdict.score
            test_case.reason = record.verdict.reason

    ##################################################
    ### Enhance attacks ##############################
    ##################################################

    def enhance_attack(
        self,
        attack: BaseAttack,
        test_case: RTTestCase,
        ignore_errors: bool,
        vulnerabilities: Optional[List[BaseVulnerability]] = None,
    ):
        from deepteam.test_case.test_case import RTTurn
        from deepteam.attacks.multi_turn import BaseMultiTurnAttack

        # Check if the attack is a multi-turn attack
        if isinstance(attack, BaseMultiTurnAttack):
            # This is a multi-turn attack
            attack_input = test_case.input
            if attack_input is None:
                return test_case

            test_case.attack_method = attack.get_name()
            turns = [RTTurn(role="user", content=attack_input)]
            metric_check, metric_holder = self._build_metric_check(
                test_case, vulnerabilities
            )

            with cost_accumulator() as acc:
                try:
                    result = attack.run(
                        model_callback=self.model_callback,
                        turns=turns,
                        vulnerability=test_case.vulnerability,
                        vulnerability_type=test_case.vulnerability_type.value,
                        simulator_model=self.simulator_model,
                        metric_check=metric_check,
                    )

                    test_case.turns = result.turns
                    test_case.actual_output = result.turns[-1].content
                    if result.error is not None:
                        raise result.error
                    self._flag_incomplete_progression(test_case)

                except ModelRefusalError as e:
                    if ignore_errors:
                        test_case.error = e.message
                        return test_case
                    else:
                        raise
                except Exception as e:
                    if ignore_errors:
                        test_case.error = (
                            f"Error enhancing multi-turn attack: {str(e)}"
                        )
                        return test_case
                    else:
                        raise
                finally:
                    test_case.simulation_cost = add_cost(
                        test_case.simulation_cost, acc[0]
                    )
                    self._adopt_metric_verdict(test_case, metric_holder)

            return test_case

        # Regular attack handling (non-multi-turn)
        attack_input = test_case.input
        if attack_input is None:
            return test_case

        test_case.attack_method = attack.get_name()
        sig = inspect.signature(attack.enhance)

        with cost_accumulator() as acc:
            try:
                res = None
                if (
                    "simulator_model" in sig.parameters
                    and "model_callback" in sig.parameters
                ):
                    res = attack.enhance(
                        attack=attack_input,
                        simulator_model=self.simulator_model,
                        model_callback=self.model_callback,
                    )
                elif "simulator_model" in sig.parameters:
                    res = attack.enhance(
                        attack=attack_input,
                        simulator_model=self.simulator_model,
                    )
                elif "model_callback" in sig.parameters:
                    res = attack.enhance(
                        attack=attack_input,
                        model_callback=self.model_callback,
                    )
                else:
                    res = attack.enhance(attack=attack_input)

                test_case.input = res

            except ModelRefusalError as e:
                if ignore_errors:
                    test_case.error = e.message
                    return test_case
                else:
                    raise
            except Exception as e:
                if ignore_errors:
                    test_case.error = (
                        f"Error enhancing regular attack: {str(e)}"
                    )
                    return test_case
                else:
                    raise
            finally:
                test_case.simulation_cost = add_cost(
                    test_case.simulation_cost, acc[0]
                )

        return test_case

    async def a_enhance_attack(
        self,
        attack: BaseAttack,
        test_case: RTTestCase,
        ignore_errors: bool,
        vulnerabilities: Optional[List[BaseVulnerability]] = None,
    ):
        from deepteam.attacks.multi_turn import (
            BaseMultiTurnAttack,
        )
        from deepteam.test_case.test_case import RTTurn

        if isinstance(attack, BaseMultiTurnAttack):
            # This is multi-turn attack
            attack_input = test_case.input
            if attack_input is None:
                return test_case

            test_case.attack_method = attack.get_name()
            sig = inspect.signature(attack.a_enhance)
            turns = [RTTurn(role="user", content=attack_input)]
            metric_check, metric_holder = self._a_build_metric_check(
                test_case, vulnerabilities
            )

            with cost_accumulator() as acc:
                try:
                    result = await attack.a_run(
                        model_callback=self.model_callback,
                        turns=turns,
                        vulnerability=test_case.vulnerability,
                        vulnerability_type=test_case.vulnerability_type.value,
                        simulator_model=self.simulator_model,
                        metric_check=metric_check,
                    )

                    test_case.turns = result.turns
                    test_case.actual_output = result.turns[-1].content
                    if result.error is not None:
                        raise result.error
                    self._flag_incomplete_progression(test_case)

                except ModelRefusalError as e:
                    if ignore_errors:
                        test_case.error = e.message
                        return test_case
                    else:
                        raise
                except Exception as e:
                    if ignore_errors:
                        test_case.error = (
                            f"Error enhancing multi-turn attack: {str(e)}"
                        )
                        return test_case
                    else:
                        raise
                finally:
                    test_case.simulation_cost = add_cost(
                        test_case.simulation_cost, acc[0]
                    )
                    self._adopt_metric_verdict(test_case, metric_holder)

            return test_case

        attack_input = test_case.input
        if attack_input is None:
            return test_case

        test_case.attack_method = attack.get_name()
        sig = inspect.signature(attack.a_enhance)

        with cost_accumulator() as acc:
            try:
                res = None
                if (
                    "simulator_model" in sig.parameters
                    and "model_callback" in sig.parameters
                ):
                    res = await attack.a_enhance(
                        attack=attack_input,
                        simulator_model=self.simulator_model,
                        model_callback=self.model_callback,
                    )
                elif "simulator_model" in sig.parameters:
                    res = await attack.a_enhance(
                        attack=attack_input,
                        simulator_model=self.simulator_model,
                    )
                elif "model_callback" in sig.parameters:
                    res = await attack.a_enhance(
                        attack=attack_input,
                        model_callback=self.model_callback,
                    )
                else:
                    res = await attack.a_enhance(attack=attack_input)

                test_case.input = res

            except ModelRefusalError as e:
                if ignore_errors:
                    test_case.error = e.message
                    return test_case
                else:
                    raise
            except Exception as e:
                if ignore_errors:
                    test_case.error = f"Error enhancing regular attack: {str(e)}"
                    return test_case
                else:
                    raise
            finally:
                test_case.simulation_cost = add_cost(
                    test_case.simulation_cost, acc[0]
                )

        return test_case
