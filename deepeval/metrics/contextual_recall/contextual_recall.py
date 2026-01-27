from typing import Optional, List, Type, Union

from deepeval.utils import (
    get_or_create_event_loop,
    prettify_list,
)
from deepeval.metrics.utils import (
    construct_verbose_logs,
    check_llm_test_case_params,
    initialize_model,
    a_generate_with_schema_and_extract,
    generate_with_schema_and_extract,
)
from deepeval.test_case import (
    LLMTestCase,
    LLMTestCaseParams,
)
from deepeval.metrics import BaseMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.contextual_recall.template import ContextualRecallTemplate
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.contextual_recall.schema import (
    ContextualRecallVerdict,
    Verdicts,
    ContextualRecallScoreReason,
    VerdictWithExpectedOutput,
)
from deepeval.metrics.api import metric_data_manager


class ContextualRecallMetric(BaseMetric):

    _required_params: List[LLMTestCaseParams] = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
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
            ContextualRecallTemplate
        ] = ContextualRecallTemplate,
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
        multimodal = test_case.multimodal

        check_llm_test_case_params(
            test_case,
            self._required_params,
            None,
            None,
            self,
            self.model,
            test_case.multimodal,
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
                expected_output = test_case.expected_output
                retrieval_context = test_case.retrieval_context

                self.verdicts: List[VerdictWithExpectedOutput] = (
                    self._generate_verdicts(
                        expected_output, retrieval_context, multimodal
                    )
                )
                self.score = self._calculate_score()
                self.reason = self._generate_reason(expected_output, multimodal)
                self.success = self.score >= self.threshold
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
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
            test_case.multimodal,
        )

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            expected_output = test_case.expected_output
            retrieval_context = test_case.retrieval_context

            self.verdicts: List[VerdictWithExpectedOutput] = (
                await self._a_generate_verdicts(
                    expected_output, retrieval_context, multimodal
                )
            )
            self.score = self._calculate_score()
            self.reason = await self._a_generate_reason(
                expected_output, multimodal
            )
            self.success = self.score >= self.threshold
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Verdicts:\n{prettify_list(self.verdicts)}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )
            if _log_metric_to_confident:
                metric_data_manager.post_metric_if_enabled(
                    self, test_case=test_case
                )
            return self.score

    async def _a_generate_reason(self, expected_output: str, multimodal: bool):
        if self.include_reason is False:
            return None

        supportive_reasons = []
        unsupportive_reasons = []
        for verdict in self.verdicts:
            if verdict.verdict.lower() == "yes":
                supportive_reasons.append(verdict.reason)
            else:
                unsupportive_reasons.append(verdict.reason)

        prompt = self.evaluation_template.generate_reason(
            expected_output=expected_output,
            supportive_reasons=supportive_reasons,
            unsupportive_reasons=unsupportive_reasons,
            score=format(self.score, ".2f"),
            multimodal=multimodal,
        )

        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=ContextualRecallScoreReason,
            extract_schema=lambda score_reason: score_reason.reason,
            extract_json=lambda data: data["reason"],
        )

    def _generate_reason(self, expected_output: str, multimodal: bool):
        if self.include_reason is False:
            return None

        supportive_reasons = []
        unsupportive_reasons = []
        for verdict in self.verdicts:
            if verdict.verdict.lower() == "yes":
                supportive_reasons.append(verdict.reason)
            else:
                unsupportive_reasons.append(verdict.reason)

        prompt = self.evaluation_template.generate_reason(
            expected_output=expected_output,
            supportive_reasons=supportive_reasons,
            unsupportive_reasons=unsupportive_reasons,
            score=format(self.score, ".2f"),
            multimodal=multimodal,
        )

        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=ContextualRecallScoreReason,
            extract_schema=lambda score_reason: score_reason.reason,
            extract_json=lambda data: data["reason"],
        )

    def _calculate_score(self):
        number_of_verdicts = len(self.verdicts)
        if number_of_verdicts == 0:
            return 0

        justified_sentences = 0
        for verdict in self.verdicts:
            if verdict.verdict.lower() == "yes":
                justified_sentences += 1

        score = justified_sentences / number_of_verdicts
        return 0 if self.strict_mode and score < self.threshold else score

    async def _a_generate_verdicts(
        self,
        expected_output: str,
        retrieval_context: List[str],
        multimodal: bool,
    ) -> List[VerdictWithExpectedOutput]:
        prompt = self.evaluation_template.generate_verdicts(
            expected_output=expected_output,
            retrieval_context=retrieval_context,
            multimodal=multimodal,
        )
        verdicts = await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=Verdicts,
            extract_schema=lambda r: list(r.verdicts),
            extract_json=lambda data: [
                ContextualRecallVerdict(**item) for item in data["verdicts"]
            ],
        )
        final_verdicts = []
        for verdict in verdicts:
            new_verdict = VerdictWithExpectedOutput(
                verdict=verdict.verdict,
                reason=verdict.reason,
                expected_output=expected_output,
            )
            final_verdicts.append(new_verdict)
        return final_verdicts

    def _generate_verdicts(
        self,
        expected_output: str,
        retrieval_context: List[str],
        multimodal: bool,
    ) -> List[VerdictWithExpectedOutput]:
        prompt = self.evaluation_template.generate_verdicts(
            expected_output=expected_output,
            retrieval_context=retrieval_context,
            multimodal=multimodal,
        )
        verdicts = generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=Verdicts,
            extract_schema=lambda r: list(r.verdicts),
            extract_json=lambda data: [
                ContextualRecallVerdict(**item) for item in data["verdicts"]
            ],
        )
        final_verdicts = []
        for verdict in verdicts:
            new_verdict = VerdictWithExpectedOutput(
                verdict=verdict.verdict,
                reason=verdict.reason,
                expected_output=expected_output,
            )
            final_verdicts.append(new_verdict)
        return final_verdicts

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
        return "Contextual Recall"
