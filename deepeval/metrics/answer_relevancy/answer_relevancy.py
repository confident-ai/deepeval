from typing import Optional, List, Type, Union

from deepeval.utils import get_or_create_event_loop, prettify_list
from deepeval.metrics.utils import (
    construct_verbose_logs,
    check_llm_test_case_params,
    a_gen_and_extract,
    gen_and_extract,
    initialize_model,
)
from deepeval.test_case import (
    LLMTestCase,
    LLMTestCaseParams,
)
from deepeval.metrics import BaseMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.answer_relevancy.template import AnswerRelevancyTemplate
from deepeval.metrics.indicator import metric_progress_indicator
import deepeval.metrics.answer_relevancy.schema as arschema
from deepeval.metrics.api import metric_data_manager


class AnswerRelevancyMetric(BaseMetric):
    _required_params: List[LLMTestCaseParams] = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ]

    def __init__(
        self,
        threshold: float = 0.5,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        evaluation_template: Type[
            AnswerRelevancyTemplate
        ] = AnswerRelevancyTemplate,
    ):
        self.threshold = 1 if strict_mode else threshold
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
                self.statements: List[str] = self._generate_statements(
                    test_case.actual_output
                )
                self.verdicts: List[arschema.AnswerRelevancyVerdict] = (
                    self._generate_verdicts(test_case.input)
                )
                self.score = self._calculate_score()
                self.reason = self._generate_reason(test_case.input)
                self.success = self.score >= self.threshold
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Statements:\n{prettify_list(self.statements)}",
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
            self.statements: List[str] = await self._a_generate_statements(
                test_case.actual_output
            )
            self.verdicts: List[arschema.AnswerRelevancyVerdict] = (
                await self._a_generate_verdicts(test_case.input)
            )
            self.score = self._calculate_score()
            self.reason = await self._a_generate_reason(test_case.input)
            self.success = self.score >= self.threshold
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Statements:\n{prettify_list(self.statements)}",
                    f"Verdicts:\n{prettify_list(self.verdicts)}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )
            if _log_metric_to_confident:
                metric_data_manager.post_metric_if_enabled(
                    self, test_case=test_case
                )
            return self.score

    async def _a_generate_reason(self, input: str) -> Optional[str]:
        if self.include_reason is False:
            return None

        irrelevant_statements = []
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "no":
                irrelevant_statements.append(verdict.reason)

        prompt = self.evaluation_template.generate_reason(
            irrelevant_statements=irrelevant_statements,
            input=input,
            score=format(self.score, ".2f"),
        )
        return await a_gen_and_extract(
            self,
            prompt,
            arschema.AnswerRelevancyScoreReason,
            extract_schema=lambda r: r.reason,
            extract_json=lambda d: d["reason"],
        )

    def _generate_reason(self, input: str) -> Optional[str]:
        if self.include_reason is False:
            return None

        irrelevant_statements = []
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "no":
                irrelevant_statements.append(verdict.reason)

        prompt = self.evaluation_template.generate_reason(
            irrelevant_statements=irrelevant_statements,
            input=input,
            score=format(self.score, ".2f"),
        )

        return gen_and_extract(
            self,
            prompt,
            arschema.AnswerRelevancyScoreReason,
            extract_schema=lambda r: r.reason,
            extract_json=lambda d: d["reason"],
        )

    async def _a_generate_verdicts(
        self, input: str
    ) -> List[arschema.AnswerRelevancyVerdict]:
        if len(self.statements) == 0:
            return []

        prompt = self.evaluation_template.generate_verdicts(
            input=input,
            statements=self.statements,
        )

        return await a_gen_and_extract(
            self,
            prompt,
            arschema.Verdicts,
            extract_schema=lambda r: list(r.verdicts),
            extract_json=lambda d: [
                arschema.AnswerRelevancyVerdict(**item)
                for item in d["verdicts"]
            ],
        )

    def _generate_verdicts(
        self, input: str
    ) -> List[arschema.AnswerRelevancyVerdict]:
        if len(self.statements) == 0:
            return []

        prompt = self.evaluation_template.generate_verdicts(
            input=input,
            statements=self.statements,
        )

        return gen_and_extract(
            self,
            prompt,
            arschema.Verdicts,
            extract_schema=lambda r: list(r.verdicts),
            extract_json=lambda d: [
                arschema.AnswerRelevancyVerdict(**item)
                for item in d["verdicts"]
            ],
        )

    async def _a_generate_statements(
        self,
        actual_output: str,
    ) -> List[str]:
        prompt = self.evaluation_template.generate_statements(
            actual_output=actual_output,
        )
        return await a_gen_and_extract(
            self,
            prompt,
            arschema.Statements,
            extract_schema=lambda r: r.statements,
            extract_json=lambda d: d["statements"],
        )

    def _generate_statements(
        self,
        actual_output: str,
    ) -> List[str]:
        prompt = self.evaluation_template.generate_statements(
            actual_output=actual_output,
        )

        return gen_and_extract(
            self,
            prompt,
            arschema.Statements,
            extract_schema=lambda r: r.statements,
            extract_json=lambda d: d["statements"],
        )

    def _calculate_score(self):
        number_of_verdicts = len(self.verdicts)
        if number_of_verdicts == 0:
            return 1

        relevant_count = 0
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() != "no":
                relevant_count += 1

        score = relevant_count / number_of_verdicts
        return 0 if self.strict_mode and score < self.threshold else score

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            try:
                self.success = self.score >= self.threshold
            except TypeError:
                self.success = False
        return self.success

    @property
    def __name__(self):
        return "Answer Relevancy"
