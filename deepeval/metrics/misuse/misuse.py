from typing import List, Optional, Type, Union

from deepeval.metrics import BaseMetric
from deepeval.test_case import (
    LLMTestCase,
    LLMTestCaseParams,
)
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.models import DeepEvalBaseLLM
from deepeval.utils import get_or_create_event_loop, prettify_list
from deepeval.metrics.utils import (
    construct_verbose_logs,
    check_llm_test_case_params,
    initialize_model,
    a_generate_with_schema_and_extract,
    generate_with_schema_and_extract,
)
from deepeval.metrics.misuse.template import MisuseTemplate
from deepeval.metrics.misuse.schema import (
    Misuses,
    MisuseVerdict,
    Verdicts,
    MisuseScoreReason,
)
from deepeval.metrics.api import metric_data_manager


class MisuseMetric(BaseMetric):
    _required_params: List[LLMTestCaseParams] = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ]

    def __init__(
        self,
        domain: str,  # Required parameter - no defaults
        threshold: float = 0.5,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        evaluation_template: Type[MisuseTemplate] = MisuseTemplate,
    ):
        if not domain or len(domain.strip()) == 0:
            raise ValueError("domain must be specified and non-empty")

        self.domain = domain.strip().lower()
        self.threshold = 0 if strict_mode else threshold
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

        multimodal = test_case.multimodal
        check_llm_test_case_params(
            test_case,
            self._required_params,
            None,
            None,
            self,
            self.model,
            multimodal,
        )

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
                self.misuses: List[str] = self._generate_misuses(
                    test_case.actual_output
                )
                self.verdicts: List[MisuseVerdict] = self._generate_verdicts()
                self.score = self._calculate_score()
                self.reason = self._generate_reason()
                self.success = self.score <= self.threshold
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Misuses:\n{prettify_list(self.misuses)}",
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

        multimodal = test_case.multimodal
        check_llm_test_case_params(
            test_case,
            self._required_params,
            None,
            None,
            self,
            self.model,
            multimodal,
        )

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            self.misuses: List[str] = await self._a_generate_misuses(
                test_case.actual_output
            )
            self.verdicts: List[MisuseVerdict] = (
                await self._a_generate_verdicts()
            )
            self.score = self._calculate_score()
            self.reason = await self._a_generate_reason()
            self.success = self.score <= self.threshold
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Misuses:\n{prettify_list(self.misuses)}",
                    f"Verdicts:\n{prettify_list(self.verdicts)}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )
            if _log_metric_to_confident:
                metric_data_manager.post_metric_if_enabled(
                    self, test_case=test_case
                )
            return self.score

    async def _a_generate_reason(self) -> Optional[str]:
        if self.include_reason is False:
            return None

        misuses = []
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "yes":
                misuses.append(verdict.reason)

        prompt: dict = self.evaluation_template.generate_reason(
            misuse_violations=misuses,
            score=format(self.score, ".2f"),
        )

        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=MisuseScoreReason,
            extract_schema=lambda s: s.reason,
            extract_json=lambda data: data["reason"],
        )

    def _generate_reason(self) -> Optional[str]:
        if self.include_reason is False:
            return None

        misuses = []
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "yes":
                misuses.append(verdict.reason)

        prompt: dict = self.evaluation_template.generate_reason(
            misuse_violations=misuses,
            score=format(self.score, ".2f"),
        )

        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=MisuseScoreReason,
            extract_schema=lambda s: s.reason,
            extract_json=lambda data: data["reason"],
        )

    async def _a_generate_verdicts(self) -> List[MisuseVerdict]:
        if len(self.misuses) == 0:
            return []

        prompt = self.evaluation_template.generate_verdicts(
            misuses=self.misuses, domain=self.domain
        )
        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=Verdicts,
            extract_schema=lambda s: list(s.verdicts),
            extract_json=lambda data: [
                MisuseVerdict(**item) for item in data["verdicts"]
            ],
        )

    def _generate_verdicts(self) -> List[MisuseVerdict]:
        if len(self.misuses) == 0:
            return []

        prompt = self.evaluation_template.generate_verdicts(
            misuses=self.misuses, domain=self.domain
        )
        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=Verdicts,
            extract_schema=lambda s: list(s.verdicts),
            extract_json=lambda data: [
                MisuseVerdict(**item) for item in data["verdicts"]
            ],
        )

    async def _a_generate_misuses(self, actual_output: str) -> List[str]:
        prompt = self.evaluation_template.generate_misuses(
            actual_output=actual_output, domain=self.domain
        )
        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=Misuses,
            extract_schema=lambda s: s.misuses,
            extract_json=lambda data: data["misuses"],
        )

    def _generate_misuses(self, actual_output: str) -> List[str]:
        prompt = self.evaluation_template.generate_misuses(
            actual_output=actual_output, domain=self.domain
        )
        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=Misuses,
            extract_schema=lambda s: s.misuses,
            extract_json=lambda data: data["misuses"],
        )

    def _calculate_score(self) -> float:
        number_of_verdicts = len(self.verdicts)
        if number_of_verdicts == 0:
            return 0

        misuse_count = 0
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "yes":
                misuse_count += 1

        score = misuse_count / number_of_verdicts
        return 1 if self.strict_mode and score > self.threshold else score

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            try:
                self.success = self.score <= self.threshold
            except TypeError:
                self.success = False
        return self.success

    @property
    def __name__(self):
        return "Misuse"
