from typing import Optional, List, Union
import asyncio

from deepeval.metrics import BaseMultimodalMetric
from deepeval.test_case import MLLMTestCaseParams, MLLMTestCase, MLLMImage
from deepeval.metrics.multimodal_metrics.multimodal_contextual_relevancy.template import (
    MultimodalContextualRelevancyTemplate,
)
from deepeval.utils import get_or_create_event_loop, prettify_list
from deepeval.metrics.utils import (
    construct_verbose_logs,
    trimAndLoadJson,
    check_mllm_test_case_params,
    initialize_multimodal_model,
)

from deepeval.models import DeepEvalBaseMLLM
from deepeval.metrics.multimodal_metrics.multimodal_contextual_relevancy.schema import *
from deepeval.metrics.indicator import metric_progress_indicator


class MultimodalContextualRelevancyMetric(BaseMultimodalMetric):

    _required_params: List[MLLMTestCaseParams] = [
        MLLMTestCaseParams.INPUT,
        MLLMTestCaseParams.ACTUAL_OUTPUT,
        MLLMTestCaseParams.RETRIEVAL_CONTEXT,
    ]

    def __init__(
        self,
        threshold: float = 0.5,
        model: Optional[Union[str, DeepEvalBaseMLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.model, self.using_native_model = initialize_multimodal_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode

    def measure(
        self,
        test_case: MLLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        check_mllm_test_case_params(
            test_case, self._required_params, None, None, self
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
                    )
                )
            else:
                self.verdicts_list: List[ContextualRelevancyVerdicts] = [
                    (self._generate_verdicts(test_case.input, context))
                    for context in test_case.retrieval_context
                ]
                self.score = self._calculate_score()
                self.reason = self._generate_reason(test_case.input)
                self.success = self.score >= self.threshold
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Verdicts:\n{prettify_list(self.verdicts_list)}",
                        f"Score: {self.score}\nReason: {self.reason}",
                    ],
                )

                return self.score

    async def a_measure(
        self,
        test_case: MLLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        check_mllm_test_case_params(
            test_case, self._required_params, None, None, self
        )

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            self.verdicts_list: List[ContextualRelevancyVerdicts] = (
                await asyncio.gather(
                    *[
                        self._a_generate_verdicts(test_case.input, context)
                        for context in test_case.retrieval_context
                    ]
                )
            )
            self.score = self._calculate_score()
            self.reason = await self._a_generate_reason(test_case.input)
            self.success = self.score >= self.threshold
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Verdicts:\n{prettify_list(self.verdicts_list)}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )

            return self.score

    async def _a_generate_reason(self, input: List[Union[str, MLLMImage]]):
        if self.include_reason is False:
            return None

        irrelevancies = []
        relevant_statements = []
        for verdicts in self.verdicts_list:
            for verdict in verdicts.verdicts:
                if verdict.verdict.lower() == "no":
                    irrelevancies.append(verdict.reason)
                else:
                    relevant_statements.append(verdict.statement)

        prompt: dict = MultimodalContextualRelevancyTemplate.generate_reason(
            input=input,
            irrelevancies=irrelevancies,
            relevant_statements=relevant_statements,
            score=format(self.score, ".2f"),
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt, schema=Reason)
            self.evaluation_cost += cost
            return res.reason
        else:
            try:
                res: Reason = await self.model.a_generate(prompt, schema=Reason)
                return res.reason
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["reason"]

    def _generate_reason(self, input: List[Union[str, MLLMImage]]):
        if self.include_reason is False:
            return None

        irrelevancies = []
        relevant_statements = []
        for verdicts in self.verdicts_list:
            for verdict in verdicts.verdicts:
                if verdict.verdict.lower() == "no":
                    irrelevancies.append(verdict.reason)
                else:
                    relevant_statements.append(verdict.statement)

        prompt: dict = MultimodalContextualRelevancyTemplate.generate_reason(
            input=input,
            irrelevancies=irrelevancies,
            relevant_statements=relevant_statements,
            score=format(self.score, ".2f"),
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt, schema=Reason)
            self.evaluation_cost += cost
            return res.reason
        else:
            try:
                res: Reason = self.model.generate(prompt, schema=Reason)
                return res.reason
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["reason"]

    def _calculate_score(self):
        total_verdicts = 0
        relevant_statements = 0
        for verdicts in self.verdicts_list:
            for verdict in verdicts.verdicts:
                total_verdicts += 1
                if verdict.verdict.lower() == "yes":
                    relevant_statements += 1

        if total_verdicts == 0:
            return 0

        score = relevant_statements / total_verdicts
        return 0 if self.strict_mode and score < self.threshold else score

    async def _a_generate_verdicts(
        self,
        input: List[Union[str, MLLMImage]],
        context: List[Union[str, MLLMImage]],
    ) -> ContextualRelevancyVerdicts:
        prompt = MultimodalContextualRelevancyTemplate.generate_verdicts(
            input=input, context=context
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(
                prompt, schema=ContextualRelevancyVerdicts
            )
            self.evaluation_cost += cost
            return res
        else:
            try:
                res = await self.model.a_generate(
                    prompt, schema=ContextualRelevancyVerdicts
                )
                return res
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return ContextualRelevancyVerdicts(**data)

    def _generate_verdicts(
        self,
        input: List[Union[str, MLLMImage]],
        context: List[Union[str, MLLMImage]],
    ) -> ContextualRelevancyVerdicts:
        prompt = MultimodalContextualRelevancyTemplate.generate_verdicts(
            input=input, context=context
        )
        if self.using_native_model:
            res, cost = self.model.generate(
                prompt, schema=ContextualRelevancyVerdicts
            )
            self.evaluation_cost += cost
            return res
        else:
            try:
                res = self.model.generate(
                    prompt, schema=ContextualRelevancyVerdicts
                )
                return res
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return ContextualRelevancyVerdicts(**data)

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
        return "Multimodal Contextual Relevancy"
