from typing import List, Optional, Type, Union

from deepeval.metrics import BaseMetric
from deepeval.test_case import (
    LLMTestCase,
    LLMTestCaseParams,
    ConversationalTestCase,
)
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.models import DeepEvalBaseLLM
from deepeval.utils import get_or_create_event_loop, prettify_list
from deepeval.metrics.utils import (
    construct_verbose_logs,
    trimAndLoadJson,
    check_llm_test_case_params,
    initialize_model,
)
from deepeval.metrics.role_violation.template import RoleViolationTemplate
from deepeval.metrics.role_violation.schema import *
from deepeval.metrics.api import metric_data_manager


class RoleViolationMetric(BaseMetric):
    _required_params: List[LLMTestCaseParams] = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ]

    def __init__(
        self,
        threshold: float = 0.5,
        role: str = None,  # Required parameter to specify the expected role
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        evaluation_template: Type[
            RoleViolationTemplate
        ] = RoleViolationTemplate,
    ):
        if role is None:
            raise ValueError(
                "Role parameter is required. Please specify the expected role (e.g., 'helpful assistant', 'customer service agent', etc.)"
            )

        self.threshold = 0 if strict_mode else threshold
        self.role = role
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.evaluation_template = evaluation_template

    def measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
        _log_metric_to_confident: bool = True,
    ) -> float:

        check_llm_test_case_params(test_case, self._required_params, self)

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self, _show_indicator=_show_indicator, _in_component=_in_component
        ):
            if self.async_mode:
                loop = get_or_create_event_loop()
                loop.run_until_complete(
                    self.a_measure(
                        test_case,
                        _show_indicator=False,
                        _in_component=_in_component,
                        _log_metric_to_confident=_log_metric_to_confident,
                    )
                )
            else:
                self.role_violations: List[str] = self._detect_role_violations(
                    test_case.actual_output
                )
                self.verdicts: List[RoleViolationVerdict] = (
                    self._generate_verdicts()
                )
                self.score = self._calculate_score()
                self.reason = self._generate_reason()
                self.success = self.score >= self.threshold
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Role: {self.role}",
                        f"Role Violations:\n{prettify_list(self.role_violations)}",
                        f"Verdicts:\n{prettify_list(self.verdicts)}",
                        f"Score: {self.score}\nReason: {self.reason}",
                    ],
                )
                if _log_metric_to_confident:
                    metric_data_manager.post_metric_if_enabled(
                        self, test_case=test_case
                    )

            return self.score

    async def a_measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
        _log_metric_to_confident: bool = True,
    ) -> float:

        check_llm_test_case_params(test_case, self._required_params, self)

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            self.role_violations: List[str] = (
                await self._a_detect_role_violations(test_case.actual_output)
            )
            self.verdicts: List[RoleViolationVerdict] = (
                await self._a_generate_verdicts()
            )
            self.score = self._calculate_score()
            self.reason = await self._a_generate_reason()
            self.success = self.score >= self.threshold
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Role: {self.role}",
                    f"Role Violations:\n{prettify_list(self.role_violations)}",
                    f"Verdicts:\n{prettify_list(self.verdicts)}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )
            if _log_metric_to_confident:
                metric_data_manager.post_metric_if_enabled(
                    self, test_case=test_case
                )

            return self.score

    async def _a_generate_reason(self) -> str:
        if self.include_reason is False:
            return None

        role_violations = []
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "yes":
                role_violations.append(verdict.reason)

        prompt: dict = self.evaluation_template.generate_reason(
            role_violations=role_violations,
            score=format(self.score, ".2f"),
        )

        if self.using_native_model:
            res, cost = await self.model.a_generate(
                prompt, schema=RoleViolationScoreReason
            )
            self.evaluation_cost += cost
            return res.reason
        else:
            try:
                res: RoleViolationScoreReason = await self.model.a_generate(
                    prompt, schema=RoleViolationScoreReason
                )
                return res.reason
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["reason"]

    def _generate_reason(self) -> str:
        if self.include_reason is False:
            return None

        role_violations = []
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "yes":
                role_violations.append(verdict.reason)

        prompt: dict = self.evaluation_template.generate_reason(
            role_violations=role_violations,
            score=format(self.score, ".2f"),
        )

        if self.using_native_model:
            res, cost = self.model.generate(
                prompt, schema=RoleViolationScoreReason
            )
            self.evaluation_cost += cost
            return res.reason
        else:
            try:
                res: RoleViolationScoreReason = self.model.generate(
                    prompt, schema=RoleViolationScoreReason
                )
                return res.reason
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["reason"]

    async def _a_generate_verdicts(self) -> List[RoleViolationVerdict]:
        if len(self.role_violations) == 0:
            return []

        verdicts: List[RoleViolationVerdict] = []
        prompt = self.evaluation_template.generate_verdicts(
            role_violations=self.role_violations
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt, schema=Verdicts)
            self.evaluation_cost += cost
            verdicts = [item for item in res.verdicts]
            return verdicts
        else:
            try:
                res: Verdicts = await self.model.a_generate(
                    prompt, schema=Verdicts
                )
                verdicts = [item for item in res.verdicts]
                return verdicts
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                verdicts = [
                    RoleViolationVerdict(**item) for item in data["verdicts"]
                ]
                return verdicts

    def _generate_verdicts(self) -> List[RoleViolationVerdict]:
        if len(self.role_violations) == 0:
            return []

        verdicts: List[RoleViolationVerdict] = []
        prompt = self.evaluation_template.generate_verdicts(
            role_violations=self.role_violations
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt, schema=Verdicts)
            self.evaluation_cost += cost
            verdicts = [item for item in res.verdicts]
            return verdicts
        else:
            try:
                res: Verdicts = self.model.generate(prompt, schema=Verdicts)
                verdicts = [item for item in res.verdicts]
                return verdicts
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                verdicts = [
                    RoleViolationVerdict(**item) for item in data["verdicts"]
                ]
                return verdicts

    async def _a_detect_role_violations(self, actual_output: str) -> List[str]:
        prompt = self.evaluation_template.detect_role_violations(
            actual_output, self.role
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(
                prompt, schema=RoleViolations
            )
            self.evaluation_cost += cost
            return res.role_violations
        else:
            try:
                res: RoleViolations = await self.model.a_generate(
                    prompt, schema=RoleViolations
                )
                return res.role_violations
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["role_violations"]

    def _detect_role_violations(self, actual_output: str) -> List[str]:
        prompt = self.evaluation_template.detect_role_violations(
            actual_output, self.role
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt, schema=RoleViolations)
            self.evaluation_cost += cost
            return res.role_violations
        else:
            try:
                res: RoleViolations = self.model.generate(
                    prompt, schema=RoleViolations
                )
                return res.role_violations
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["role_violations"]

    def _calculate_score(self) -> float:
        # Role adherence should be binary: either there's adherence (1) or not (0)
        number_of_verdicts = len(self.verdicts)
        if number_of_verdicts == 0:
            return 1

        # If any verdict indicates a role violation, score is 0, otherwise 1
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "yes":
                return 0.0  # Role violation detected - no adherence
        return 1.0  # No role violation - full adherence

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            try:
                self.success = self.score >= self.threshold
            except:
                self.success = False
        return self.success

    @property
    def __name__(self):
        return "Role Violation"
