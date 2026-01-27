import asyncio

from typing import Optional, List, Union

from deepeval.utils import (
    get_or_create_event_loop,
    prettify_list,
    get_per_task_timeout,
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
from deepeval.metrics.prompt_alignment.template import PromptAlignmentTemplate
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.prompt_alignment import schema as paschema

from deepeval.metrics.api import metric_data_manager


class PromptAlignmentMetric(BaseMetric):

    _required_params: List[LLMTestCaseParams] = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ]

    def __init__(
        self,
        prompt_instructions: List[str],
        threshold: float = 0.5,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
    ):
        if len(prompt_instructions) == 0:
            raise ValueError("'prompt_instructions' must not be empty.")

        self.prompt_instructions = prompt_instructions
        self.threshold = 1 if strict_mode else threshold
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode

    def measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
        _log_metric_to_confident: bool = True,
    ) -> float:

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
                coro = self.a_measure(
                    test_case,
                    _show_indicator=False,
                    _in_component=_in_component,
                    _log_metric_to_confident=_log_metric_to_confident,
                )
                loop.run_until_complete(
                    asyncio.wait_for(
                        coro,
                        timeout=get_per_task_timeout(),
                    )
                )
            else:
                self.verdicts: List[paschema.PromptAlignmentVerdict] = (
                    self._generate_verdicts(
                        test_case.input, test_case.actual_output
                    )
                )
                self.score = self._calculate_score()
                self.reason = self._generate_reason(
                    test_case.input, test_case.actual_output
                )
                self.success = self.score >= self.threshold
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Prompt Instructions:\n{prettify_list(self.prompt_instructions)}",
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
            self.verdicts: List[paschema.PromptAlignmentVerdict] = (
                await self._a_generate_verdicts(
                    test_case.input, test_case.actual_output
                )
            )
            self.score = self._calculate_score()
            self.reason = await self._a_generate_reason(
                test_case.input, test_case.actual_output
            )
            self.success = self.score >= self.threshold
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Prompt Instructions:\n{prettify_list(self.prompt_instructions)}",
                    f"Verdicts:\n{prettify_list(self.verdicts)}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )
            if _log_metric_to_confident:
                metric_data_manager.post_metric_if_enabled(
                    self, test_case=test_case
                )
            return self.score

    async def _a_generate_reason(
        self, input: str, actual_output: str
    ) -> Optional[str]:
        if self.include_reason is False:
            return None

        unalignment_reasons = []
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "no":
                unalignment_reasons.append(verdict.reason)

        prompt = PromptAlignmentTemplate.generate_reason(
            unalignment_reasons=unalignment_reasons,
            input=input,
            actual_output=actual_output,
            score=format(self.score, ".2f"),
        )

        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=paschema.PromptAlignmentScoreReason,
            extract_schema=lambda s: s.reason,
            extract_json=lambda data: data["reason"],
        )

    def _generate_reason(self, input: str, actual_output: str) -> Optional[str]:
        if self.include_reason is False:
            return None

        unalignment_reasons = []
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "no":
                unalignment_reasons.append(verdict.reason)

        prompt = PromptAlignmentTemplate.generate_reason(
            unalignment_reasons=unalignment_reasons,
            input=input,
            actual_output=actual_output,
            score=format(self.score, ".2f"),
        )

        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=paschema.PromptAlignmentScoreReason,
            extract_schema=lambda s: s.reason,
            extract_json=lambda data: data["reason"],
        )

    async def _a_generate_verdicts(
        self, input: str, actual_output: str
    ) -> List[paschema.PromptAlignmentVerdict]:
        prompt = PromptAlignmentTemplate.generate_verdicts(
            prompt_instructions=self.prompt_instructions,
            input=input,
            actual_output=actual_output,
        )
        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=paschema.Verdicts,
            extract_schema=lambda s: list(s.verdicts),
            extract_json=lambda data: [
                paschema.PromptAlignmentVerdict(**item)
                for item in data["verdicts"]
            ],
        )

    def _generate_verdicts(
        self, input: str, actual_output: str
    ) -> List[paschema.PromptAlignmentVerdict]:
        prompt = PromptAlignmentTemplate.generate_verdicts(
            prompt_instructions=self.prompt_instructions,
            input=input,
            actual_output=actual_output,
        )
        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=paschema.Verdicts,
            extract_schema=lambda s: list(s.verdicts),
            extract_json=lambda data: [
                paschema.PromptAlignmentVerdict(**item)
                for item in data["verdicts"]
            ],
        )

    def _calculate_score(self) -> float:
        number_of_verdicts = len(self.verdicts)
        if number_of_verdicts == 0:
            return 1

        alignment_count = 0
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() != "no":
                alignment_count += 1

        score = alignment_count / number_of_verdicts
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
        return "Prompt Alignment"
