from typing import List, Literal, Optional, Union, Dict
from enum import Enum
import asyncio

from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.utils import initialize_model, trimAndLoadJson
from deepeval.utils import get_or_create_event_loop

from deepteam.utils import validate_model_callback_signature
from deepteam.confident.api import Api, Endpoints
from deepeval.confident.api import HttpMethods
from rich.console import Console

from deepteam.vulnerabilities import BaseVulnerability
from deepteam.metrics import BaseRedTeamingMetric, HarmMetric
from deepteam.metrics.types import EvaluationExample
from deepteam.attacks.multi_turn.types import CallbackType
from deepteam.attacks.attack_engine import AttackEngine
from deepteam.attacks.attack_simulator.schema import SyntheticDataList
from deepteam.risks import getRiskCategory
from deepteam.test_case import RTTestCase
from .template import CustomVulnerabilityTemplate
from .api import (
    APIEvaluationExample,
    CustomVulnerabilityHttpResponse,
    CustomVulnerabilityUploadRequest,
)
from deepeval.tracing.types import Trace
from deepteam.trace_scanner.schema import BatchFinding
from deepteam.trace_scanner import TraceScanner


def _build_types(name: str, types: Optional[List[str]]) -> List[Enum]:
    if types:
        vulnerability_types = Enum("CustomVulnerabilityType", {t.upper(): t for t in types})
    else:
        vulnerability_types = Enum(
            "CustomVulnerabilityType", {name.upper().replace(" ", "_"): name}
        )
    return list(vulnerability_types)


class CustomVulnerability(BaseVulnerability):
    """
    Custom vulnerability class that allows users to define their own vulnerability types.
    """

    def __init__(
        self,
        name: str,
        criteria: Optional[str] = None,
        types: Optional[List[str]] = None,
        custom_prompt: Optional[str] = None,
        async_mode: bool = True,
        verbose_mode: bool = False,
        simulator_model: Optional[
            Union[str, DeepEvalBaseLLM]
        ] = "gpt-3.5-turbo-0125",
        evaluation_model: Optional[Union[str, DeepEvalBaseLLM]] = "gpt-4o",
        evaluation_examples: Optional[List[EvaluationExample]] = None,
        evaluation_guidelines: Optional[List[str]] = None,
        attack_engine: Optional[AttackEngine] = None,
    ):
        self.name = name

        self.types = _build_types(name, types)

        self.custom_prompt = custom_prompt
        self.criteria = criteria.strip() if criteria else None
        self.simulator_model = simulator_model
        self.evaluation_model = evaluation_model
        self.evaluation_examples = evaluation_examples
        self.evaluation_guidelines = evaluation_guidelines
        self.async_mode = async_mode
        self.verbose_mode = verbose_mode
        self.metric = None
        super().__init__(self.types)
        self.attack_engine = attack_engine

    def get_name(self) -> str:
        return self.name

    def get_custom_prompt(self) -> Optional[str]:
        return self.custom_prompt

    def _ensure_criteria(self) -> str:
        if not self.criteria:
            raise ValueError(
                f"Custom vulnerability '{self.name}' has no criteria. Pass criteria= when "
                "constructing it, or call pull() to load it from Confident AI."
            )
        return self.criteria

    def assess(
        self,
        model_callback: CallbackType,
        purpose: Optional[str] = None,
    ) -> Dict[Enum, List[RTTestCase]]:

        validate_model_callback_signature(
            model_callback=model_callback,
            async_mode=self.async_mode,
        )

        if self.async_mode:
            loop = get_or_create_event_loop()
            return loop.run_until_complete(
                self.a_assess(
                    model_callback=model_callback,
                    purpose=purpose,
                )
            )

        simulated_test_cases = self.simulate_attacks(purpose)

        results: Dict[Enum, List[RTTestCase]] = {}
        res: Dict[Enum, BaseRedTeamingMetric] = {}
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

        self.res = res
        self.simulated_attacks = simulated_attacks

        return results

    async def a_assess(
        self,
        model_callback: CallbackType,
        purpose: Optional[str] = None,
    ) -> Dict[Enum, List[RTTestCase]]:
        validate_model_callback_signature(
            model_callback=model_callback,
            async_mode=self.async_mode,
        )

        simulated_test_cases = await self.a_simulate_attacks(purpose)

        results: Dict[Enum, List[RTTestCase]] = {}
        res: Dict[Enum, BaseRedTeamingMetric] = {}
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

        tasks = [
            process_attack(test_case)
            for test_case in simulated_test_cases
            if test_case.vulnerability_type in self.types
        ]

        for coro in asyncio.as_completed(tasks):
            vuln_type, test_case = await coro
            results.setdefault(vuln_type, []).append(test_case)

        self.res = res
        self.simulated_attacks = simulated_attacks

        return results

    def simulate_attacks(
        self,
        purpose: str = None,
        attacks_per_vulnerability_type: int = 1,
    ) -> List[RTTestCase]:
        self._ensure_criteria()

        self.purpose = purpose

        self.simulator_model, self.using_native_model = initialize_model(
            self.simulator_model
        )

        templates = dict()
        simulated_test_cases: List[RTTestCase] = []

        for type in self.types:
            templates[type] = templates.get(type, [])
            templates[type].append(
                CustomVulnerabilityTemplate.generate_baseline_attacks(
                    self.name,
                    type,
                    attacks_per_vulnerability_type,
                    self.custom_prompt,
                    self.purpose,
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
        purpose: str = None,
        attacks_per_vulnerability_type: int = 1,
    ) -> List[RTTestCase]:
        self._ensure_criteria()

        self.purpose = purpose

        self.simulator_model, self.using_native_model = initialize_model(
            self.simulator_model
        )

        templates = dict()
        simulated_test_cases: List[RTTestCase] = []

        for type in self.types:
            templates[type] = templates.get(type, [])
            templates[type].append(
                CustomVulnerabilityTemplate.generate_baseline_attacks(
                    self.name,
                    type,
                    attacks_per_vulnerability_type,
                    self.custom_prompt,
                    self.purpose,
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

    def _get_trace_template(self):
        return (
            CustomVulnerabilityTemplate.custom_vulnerability_template_wrapper(
                name=self.name,
                criteria=self._ensure_criteria(),
                type_values=self.get_values(),
            )
        )

    def _assess_trace(
        self,
        trace: Trace,
        previous_detections: Optional[List[BatchFinding]] = None,
    ) -> List[BatchFinding]:
        """
        Evaluates an entire execution trace for custom vulnerabilities using bottoms-up batching.
        """
        if self.async_mode:
            loop = get_or_create_event_loop()
            return loop.run_until_complete(self._a_assess_trace(trace=trace, previous_detections=previous_detections))

        self.evaluation_model, self.using_native_model = initialize_model(
            self.evaluation_model
        )
        trace_scanner = TraceScanner(
            model=self.evaluation_model,
            template=self._get_trace_template(),
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
        Asynchronously evaluates an entire execution trace for custom vulnerabilities.
        """
        self.evaluation_model, self.using_native_model = initialize_model(
            self.evaluation_model
        )

        trace_scanner = TraceScanner(
            model=self.evaluation_model,
            template=self._get_trace_template(),
            previous_detections=previous_detections,
        )

        findings = await trace_scanner.a_process_trace(trace)

        self.trace_findings = findings
        self.vulnerable = any(f.outcome == "materialized" for f in findings)

        return findings

    def _get_metric(self, type: Enum) -> BaseRedTeamingMetric:
        if self.metric is None:
            self.metric = HarmMetric(
                harm_category=self._ensure_criteria(),
                model=self.evaluation_model,
                async_mode=self.async_mode,
                verbose_mode=self.verbose_mode,
                evaluation_examples=self.evaluation_examples,
                evaluation_guidelines=self.evaluation_guidelines,
            )
        return self.metric

    def is_vulnerable(self) -> bool:
        self.vulnerable = False
        try:
            for _, metric_data in self.res.items():
                if metric_data.score < 1:
                    self.vulnerable = True
        except:
            self.vulnerable = False
        return self.vulnerable

    def get_criteria(self) -> Optional[str]:
        return self.criteria

    def upload(self) -> str:
        request = CustomVulnerabilityUploadRequest(
            name=self.name,
            criteria=self._ensure_criteria(),
            vulnerabilityTypes=self.get_values(),
            evaluationGuidelines=self.evaluation_guidelines,
            evaluationExamples=(
                [
                    APIEvaluationExample(
                        input=example.input,
                        actualOutput=example.actual_output,
                        score=example.score,
                        reason=example.reason,
                    )
                    for example in self.evaluation_examples
                ]
                if self.evaluation_examples is not None
                else None
            ),
        )

        try:
            body = request.model_dump(by_alias=True, exclude_none=True)
        except AttributeError:
            body = request.dict(by_alias=True, exclude_none=True)

        api = Api()
        data, _ = api.send_request(
            method=HttpMethods.POST,
            endpoint=Endpoints.VULNERABILITIES_ENDPOINT,
            body=body,
        )

        self.vulnerability_id = data["id"]
        Console().print(
            "[rgb(5,245,141)]✓[/rgb(5,245,141)] Vulnerability "
            f"'{self.name}' uploaded successfully "
            f"(id: [bold]{self.vulnerability_id}[/bold])"
        )
        return self.vulnerability_id

    def pull(self) -> None:
        api = Api()
        data, _ = api.send_request(
            method=HttpMethods.GET,
            endpoint=Endpoints.VULNERABILITY_ENDPOINT,
            url_params={"vulnerabilityId": self.name},
        )

        response = CustomVulnerabilityHttpResponse(**data["vulnerability"])

        if response.built_in:
            raise ValueError(
                f"'{response.name}' is a built-in vulnerability, not a custom one. "
                "Import it from deepteam.vulnerabilities instead of pulling it."
            )
        if not response.criteria:
            raise ValueError(
                f"Custom vulnerability '{response.name}' has no criteria on Confident AI, "
                "so there is nothing to red team it against. Set its criteria and pull again."
            )

        self.name = response.name
        self.criteria = response.criteria.strip()
        self.types = _build_types(
            response.name, [t.name for t in response.vulnerability_types]
        )
        self.evaluation_guidelines = response.evaluation_guidelines or None
        self.evaluation_examples = [
            EvaluationExample(
                input=example.input,
                actual_output=example.actual_output,
                score=example.score,
                reason=example.reason,
            )
            for example in response.evaluation_examples
        ] or None
        self.metric = None
        self.vulnerability_id = response.id

        Console().print(
            "[rgb(5,245,141)]✓[/rgb(5,245,141)] Vulnerability '{self.name}' pulled successfully"
        )
