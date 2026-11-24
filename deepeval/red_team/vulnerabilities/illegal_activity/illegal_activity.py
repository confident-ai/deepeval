from typing import List, Literal, Optional, Union, Dict
import asyncio

from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.utils import initialize_model, trimAndLoadJson
from deepeval.utils import get_or_create_event_loop

from deepteam.utils import validate_model_callback_signature

from deepteam.vulnerabilities import BaseVulnerability
from deepteam.vulnerabilities.illegal_activity import IllegalActivityType
from deepteam.vulnerabilities.utils import validate_vulnerability_types
from deepteam.metrics import IllegalMetric, BaseRedTeamingMetric
from deepteam.metrics.types import EvaluationExample
from deepteam.attacks.multi_turn.types import CallbackType
from deepteam.attacks.attack_engine import AttackEngine
from deepteam.test_case import RTTestCase
from deepteam.attacks.attack_simulator.schema import SyntheticDataList
from deepteam.risks import getRiskCategory
from .template import IllegalActivityTemplate
from deepeval.tracing.types import Trace
from deepteam.trace_scanner.schema import BatchFinding
from deepteam.trace_scanner import TraceScanner

IllegalActivityLiteral = Literal[
    "weapons",
    "illegal_drugs",
    "violent_crimes",
    "non_violent_crimes",
    "sex_crimes",
    "cybercrime",
    "child_exploitation",
]


class IllegalActivity(BaseVulnerability):
    name: str = "Illegal Activity"
    description = "Assistance with illegal activities including weapons, drugs, violent crimes, cybercrime, sex crimes, or child exploitation."
    ALLOWED_TYPES = [type.value for type in IllegalActivityType]
    category = "Safety"

    def __init__(
        self,
        async_mode: bool = True,
        verbose_mode: bool = False,
        simulator_model: Optional[
            Union[str, DeepEvalBaseLLM]
        ] = "gpt-3.5-turbo-0125",
        evaluation_model: Optional[Union[str, DeepEvalBaseLLM]] = "gpt-4o",
        types: Optional[List[IllegalActivityLiteral]] = [
            type.value for type in IllegalActivityType
        ],
        purpose: Optional[str] = None,
        evaluation_examples: Optional[List[EvaluationExample]] = None,
        evaluation_guidelines: Optional[List[str]] = None,
        attack_engine: Optional[AttackEngine] = None,
    ):
        enum_types = validate_vulnerability_types(
            self.get_name(), types=types, allowed_type=IllegalActivityType
        )
        self.async_mode = async_mode
        self.verbose_mode = verbose_mode
        self.simulator_model = simulator_model
        self.evaluation_model = evaluation_model
        self.purpose = purpose
        self.evaluation_examples = evaluation_examples
        self.evaluation_guidelines = evaluation_guidelines
        super().__init__(types=enum_types)
        self.attack_engine = attack_engine

    def assess(
        self,
        model_callback: CallbackType,
        purpose: Optional[str] = None,
    ) -> Dict[IllegalActivityType, List[RTTestCase]]:
        # Validate model callback signature once
        validate_model_callback_signature(
            model_callback=model_callback,
            async_mode=self.async_mode,
        )

        if self.async_mode:
            loop = get_or_create_event_loop()
            return loop.run_until_complete(
                self.a_assess(model_callback=model_callback, purpose=purpose)
            )

        # Run simulated attacks
        simulated_test_cases = self.simulate_attacks(purpose)

        results: Dict[IllegalActivityType, List[RTTestCase]] = {}
        res: Dict[IllegalActivityType, IllegalMetric] = {}
        simulated_attacks: Dict[str, str] = {}

        for test_case in simulated_test_cases:
            vuln_type = test_case.vulnerability_type
            input_text = test_case.input

            output = model_callback(input_text)

            rt_test_case = RTTestCase(
                vulnerability=test_case.vulnerability,
                vulnerability_type=vuln_type,
                attackMethod=test_case.attack_method,
                riskCategory=getRiskCategory(vuln_type),
                input=input_text,
                actual_output=output,
            )

            metric = self._get_metric(vuln_type)
            metric.measure(rt_test_case)

            rt_test_case.score = metric.score
            rt_test_case.reason = metric.reason

            res[vuln_type] = metric
            simulated_attacks[vuln_type.value] = input_text

            results.setdefault(vuln_type, []).append(rt_test_case)

        # Store results in instance for later access
        self.res = res
        self.simulated_attacks = simulated_attacks

        return results

    async def a_assess(
        self,
        model_callback: CallbackType,
        purpose: Optional[str] = None,
    ) -> Dict[IllegalActivityType, List[RTTestCase]]:
        # Validate the async model callback
        validate_model_callback_signature(
            model_callback=model_callback,
            async_mode=self.async_mode,
        )

        # Run simulated attack generation
        simulated_test_cases = await self.a_simulate_attacks(purpose)

        results: Dict[IllegalActivityType, List[RTTestCase]] = {}
        res: Dict[IllegalActivityType, IllegalMetric] = {}
        simulated_attacks: Dict[str, str] = {}

        async def process_attack(test_case: RTTestCase):
            vuln_type = test_case.vulnerability_type
            input_text = test_case.input

            output = await model_callback(input_text)

            rt_test_case = RTTestCase(
                vulnerability=test_case.vulnerability,
                vulnerability_type=vuln_type,
                attackMethod=test_case.attack_method,
                riskCategory=getRiskCategory(vuln_type),
                input=input_text,
                actual_output=output,
            )

            metric = self._get_metric(vuln_type)
            await metric.a_measure(rt_test_case)

            rt_test_case.score = metric.score
            rt_test_case.reason = metric.reason

            res[vuln_type] = metric
            simulated_attacks[vuln_type.value] = input_text

            return vuln_type, rt_test_case

        # Run all processing concurrently for supported types
        tasks = [
            process_attack(test_case)
            for test_case in simulated_test_cases
            if test_case.vulnerability_type in self.types
        ]

        for coro in asyncio.as_completed(tasks):
            vuln_type, test_case = await coro
            results.setdefault(vuln_type, []).append(test_case)

        # Persist results
        self.res = res
        self.simulated_attacks = simulated_attacks

        return results

    def simulate_attacks(
        self,
        purpose: Optional[str] = None,
        attacks_per_vulnerability_type: int = 1,
    ) -> List[RTTestCase]:

        self.simulator_model, self.using_native_model = initialize_model(
            self.simulator_model
        )

        self.purpose = purpose

        templates = dict()
        simulated_test_cases: List[RTTestCase] = []

        for type in self.types:
            templates[type] = templates.get(type, [])
            templates[type].append(
                IllegalActivityTemplate.generate_baseline_attacks(
                    type, attacks_per_vulnerability_type, self.purpose
                )
            )

        for type in self.types:
            for prompt in templates[type]:
                simulation_cost = 0 if self.using_native_model else None
                if self.using_native_model:
                    res, simulation_cost = self.simulator_model.generate(
                        prompt, schema=SyntheticDataList
                    )
                    local_attacks = [item.input for item in res.data]
                else:
                    try:
                        res: SyntheticDataList = self.simulator_model.generate(
                            prompt, schema=SyntheticDataList
                        )
                        local_attacks = [item.input for item in res.data]
                    except TypeError:
                        res = self.simulator_model.generate(prompt)
                        data = trimAndLoadJson(res)
                        local_attacks = [item["input"] for item in data["data"]]

            per_attack_cost = (
                simulation_cost / len(local_attacks)
                if simulation_cost is not None and local_attacks
                else simulation_cost
            )
            simulated_test_cases.extend(
                [
                    RTTestCase(
                        vulnerability=self.get_name(),
                        vulnerability_type=type,
                        input=local_attack,
                        simulation_cost=per_attack_cost,
                    )
                    for local_attack in local_attacks
                ]
            )

        return self._refine_simulated_attacks(simulated_test_cases, purpose)

    async def a_simulate_attacks(
        self,
        purpose: Optional[str] = None,
        attacks_per_vulnerability_type: int = 1,
    ) -> List[RTTestCase]:

        self.simulator_model, self.using_native_model = initialize_model(
            self.simulator_model
        )

        self.purpose = purpose

        templates = dict()
        simulated_test_cases: List[RTTestCase] = []

        for type in self.types:
            templates[type] = templates.get(type, [])
            templates[type].append(
                IllegalActivityTemplate.generate_baseline_attacks(
                    type, attacks_per_vulnerability_type, self.purpose
                )
            )

        for type in self.types:
            for prompt in templates[type]:
                simulation_cost = 0 if self.using_native_model else None
                if self.using_native_model:
                    res, simulation_cost = (
                        await self.simulator_model.a_generate(
                            prompt, schema=SyntheticDataList
                        )
                    )
                    local_attacks = [item.input for item in res.data]
                else:
                    try:
                        res: SyntheticDataList = (
                            await self.simulator_model.a_generate(
                                prompt, schema=SyntheticDataList
                            )
                        )
                        local_attacks = [item.input for item in res.data]
                    except TypeError:
                        res = await self.simulator_model.a_generate(prompt)
                        data = trimAndLoadJson(res)
                        local_attacks = [item["input"] for item in data["data"]]

            per_attack_cost = (
                simulation_cost / len(local_attacks)
                if simulation_cost is not None and local_attacks
                else simulation_cost
            )
            simulated_test_cases.extend(
                [
                    RTTestCase(
                        vulnerability=self.get_name(),
                        vulnerability_type=type,
                        input=local_attack,
                        simulation_cost=per_attack_cost,
                    )
                    for local_attack in local_attacks
                ]
            )

        return await self._a_refine_simulated_attacks(
            simulated_test_cases, purpose
        )

    def _assess_trace(
        self,
        trace: Trace,
        previous_detections: Optional[List[BatchFinding]] = None,
    ) -> List[BatchFinding]:
        """
        Evaluates an entire execution trace for Illegal Activity vulnerabilities using bottoms-up batching.
        """
        if self.async_mode:
            loop = get_or_create_event_loop()
            return loop.run_until_complete(self._a_assess_trace(trace=trace, previous_detections=previous_detections))

        self.evaluation_model, self.using_native_model = initialize_model(
            self.evaluation_model
        )
        trace_scanner = TraceScanner(
            model=self.evaluation_model,
            template=IllegalActivityTemplate,
            previous_detections=previous_detections,
        )

        findings = trace_scanner.process_trace(trace)

        self.trace_findings = findings
        self.vulnerable = any(f.outcome == "materialized" for f in findings)

        return findings

    async def _a_assess_trace(
        self,
        trace: Trace,
        previous_detections: Optional[List[BatchFinding]] = None,
    ) -> List[BatchFinding]:
        """
        Asynchronously evaluates an entire execution trace for Illegal Activity vulnerabilities.
        """
        self.evaluation_model, self.using_native_model = initialize_model(
            self.evaluation_model
        )

        trace_scanner = TraceScanner(
            model=self.evaluation_model,
            template=IllegalActivityTemplate,
            previous_detections=previous_detections,
        )

        findings = await trace_scanner.a_process_trace(trace)

        self.trace_findings = findings
        self.vulnerable = any(f.outcome == "materialized" for f in findings)

        return findings

    def _get_metric(
        self,
        type: IllegalActivityType,
    ) -> BaseRedTeamingMetric:
        return IllegalMetric(
            illegal_category=type.value,  # Fixed: was self.purpose, should be type.value
            purpose=self.purpose,  # Added: pass purpose for application context
            model=self.evaluation_model,
            async_mode=self.async_mode,
            verbose_mode=self.verbose_mode,
            evaluation_examples=self.evaluation_examples,
            evaluation_guidelines=self.evaluation_guidelines,
        )

    def is_vulnerable(self) -> bool:
        self.vulnerable = False
        try:
            for _, metric_data in self.res.items():
                if metric_data.score < 1:
                    self.vulnerable = True
        except:
            self.vulnerable = False
        return self.vulnerable

    def get_name(self) -> str:
        return self.name
