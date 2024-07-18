from typing import Optional, List, Union
import asyncio

from deepeval.utils import get_or_create_event_loop, prettify_list
from deepeval.metrics.utils import (
    construct_verbose_logs,
    validate_conversational_test_case,
    trimAndLoadJson,
    check_llm_test_case_params,
    initialize_model,
)
from deepeval.test_case import (
    LLMTestCase,
    LLMTestCaseParams,
    ConversationalTestCase,
)
from deepeval.metrics import BaseMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.contextual_relevancy.template import (
    ContextualRelevancyTemplate,
)
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.contextual_relevancy.schema import *

required_params: List[LLMTestCaseParams] = [
    LLMTestCaseParams.INPUT,
    LLMTestCaseParams.ACTUAL_OUTPUT,
    LLMTestCaseParams.RETRIEVAL_CONTEXT,
]


class ContextualRelevancyMetric(BaseMetric):
    def __init__(
        self,
        threshold: float = 0.5,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode

    def measure(
        self, test_case: Union[LLMTestCase, ConversationalTestCase]
    ) -> float:
        if isinstance(test_case, ConversationalTestCase):
            test_case = validate_conversational_test_case(test_case, self)
        check_llm_test_case_params(test_case, required_params, self)

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(self):
            if self.async_mode:
                loop = get_or_create_event_loop()
                loop.run_until_complete(
                    self.a_measure(test_case, _show_indicator=False)
                )
            else:
                self.verdicts: List[ContextualRelevancyVerdict] = (
                    self._generate_verdicts(
                        test_case.input, test_case.retrieval_context
                    )
                )
                self.score = self._calculate_score()
                self.reason = self._generate_reason(test_case.input)
                self.success = self.score >= self.threshold
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Verdicts:\n{prettify_list(self.verdicts)}",
                        f"Score: {self.score}\nReason: {self.reason}",
                    ],
                )

                return self.score

    async def a_measure(
        self,
        test_case: Union[LLMTestCase, ConversationalTestCase],
        _show_indicator: bool = True,
    ) -> float:
        if isinstance(test_case, ConversationalTestCase):
            test_case = validate_conversational_test_case(test_case, self)
        check_llm_test_case_params(test_case, required_params, self)

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
        ):
            self.verdicts: List[ContextualRelevancyVerdict] = (
                await self._a_generate_verdicts(
                    test_case.input, test_case.retrieval_context
                )
            )
            self.score = self._calculate_score()
            self.reason = await self._a_generate_reason(test_case.input)
            self.success = self.score >= self.threshold
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Verdicts:\n{prettify_list(self.verdicts)}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )

            return self.score

    async def _a_generate_reason(self, input: str):
        if self.include_reason is False:
            return None

        irrelevancies = []
        for verdict in self.verdicts:
            if verdict.verdict.lower() == "no":
                irrelevancies.append(verdict.reason)

        prompt: dict = ContextualRelevancyTemplate.generate_reason(
            input=input,
            irrelevancies=irrelevancies,
            score=format(self.score, ".2f"),
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return data["reason"]
        else:
            try:
                res: Reason = await self.model.a_generate(prompt, schema=Reason)
                return res.reason
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["reason"]

    def _generate_reason(self, input: str):
        if self.include_reason is False:
            return None

        irrelevancies = []
        for verdict in self.verdicts:
            if verdict.verdict.lower() == "no":
                irrelevancies.append(verdict.reason)

        prompt: dict = ContextualRelevancyTemplate.generate_reason(
            input=input,
            irrelevancies=irrelevancies,
            score=format(self.score, ".2f"),
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return data["reason"]
        else:
            try:
                res: Reason = self.model.generate(prompt, schema=Reason)
                return res.reason
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["reason"]

    def _calculate_score(self):
        total_verdicts = len(self.verdicts)
        if total_verdicts == 0:
            return 0

        relevant_nodes = 0
        for verdict in self.verdicts:
            if verdict.verdict.lower() == "yes":
                relevant_nodes += 1

        score = relevant_nodes / total_verdicts
        return 0 if self.strict_mode and score < self.threshold else score

    async def _a_generate_verdict(
        self, prompt: str
    ) -> ContextualRelevancyVerdict:
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return ContextualRelevancyVerdict(**data)
        else:
            try:
                res = await self.model.a_generate(
                    prompt, schema=ContextualRelevancyVerdict
                )
                return res
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return ContextualRelevancyVerdict(**data)

    async def _a_generate_verdicts(
        self, text: str, retrieval_context: List[str]
    ) -> ContextualRelevancyVerdict:
        tasks = []
        for context in retrieval_context:
            prompt = ContextualRelevancyTemplate.generate_verdict(
                text=text, context=context
            )
            task = asyncio.create_task(self._a_generate_verdict(prompt))
            tasks.append(task)
        verdicts = await asyncio.gather(*tasks)
        return verdicts

    def _generate_verdicts(
        self, text: str, retrieval_context: List[str]
    ) -> List[ContextualRelevancyVerdict]:
        verdicts: List[ContextualRelevancyVerdict] = []
        for context in retrieval_context:
            prompt = ContextualRelevancyTemplate.generate_verdict(
                text=text, context=context
            )
            if self.using_native_model:
                res, cost = self.model.generate(prompt)
                self.evaluation_cost += cost
                data = trimAndLoadJson(res, self)
                verdict = ContextualRelevancyVerdict(**data)
            else:
                try:
                    res = self.model.generate(
                        prompt, schema=ContextualRelevancyVerdict
                    )
                    verdict = res
                except TypeError:
                    res = self.model.generate(prompt)
                    data = trimAndLoadJson(res, self)
                    verdict = ContextualRelevancyVerdict(**data)

            verdicts.append(verdict)

        return verdicts

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
        return "Contextual Relevancy"
