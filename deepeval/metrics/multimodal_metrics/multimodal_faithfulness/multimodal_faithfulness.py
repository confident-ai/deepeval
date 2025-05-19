from typing import List, Optional, Union
import asyncio

from deepeval.metrics import BaseMultimodalMetric
from deepeval.test_case import MLLMTestCaseParams, MLLMTestCase, MLLMImage
from deepeval.metrics.multimodal_metrics.multimodal_faithfulness.template import (
    MultimodalFaithfulnessTemplate,
)
from deepeval.utils import get_or_create_event_loop, prettify_list
from deepeval.metrics.utils import (
    construct_verbose_logs,
    trimAndLoadJson,
    check_mllm_test_case_params,
    initialize_multimodal_model,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.multimodal_metrics.multimodal_faithfulness.schema import *
from deepeval.metrics.indicator import metric_progress_indicator


class MultimodalFaithfulnessMetric(BaseMultimodalMetric):

    _required_params: List[MLLMTestCaseParams] = [
        MLLMTestCaseParams.INPUT,
        MLLMTestCaseParams.ACTUAL_OUTPUT,
        MLLMTestCaseParams.RETRIEVAL_CONTEXT,
    ]

    def __init__(
        self,
        threshold: float = 0.5,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        truths_extraction_limit: Optional[int] = None,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.model, self.using_native_model = initialize_multimodal_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode

        self.truths_extraction_limit = truths_extraction_limit
        if self.truths_extraction_limit is not None:
            self.truths_extraction_limit = max(self.truths_extraction_limit, 0)

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
            self,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
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
                self.truths = self._generate_truths(test_case.retrieval_context)
                self.claims = self._generate_claims(test_case.actual_output)
                self.verdicts = self._generate_verdicts()
                self.score = self._calculate_score()
                self.reason = self._generate_reason()
                self.success = self.score >= self.threshold
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Truths (limit={self.truths_extraction_limit}):\n{prettify_list(self.truths)}",
                        f"Claims:\n{prettify_list(self.claims)}",
                        f"Verdicts:\n{prettify_list(self.verdicts)}",
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
            self.truths, self.claims = await asyncio.gather(
                self._a_generate_truths(test_case.retrieval_context),
                self._a_generate_claims(test_case.actual_output),
            )
            self.verdicts = await self._a_generate_verdicts()
            self.score = self._calculate_score()
            self.reason = await self._a_generate_reason()
            self.success = self.score >= self.threshold
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Truths (limit={self.truths_extraction_limit}):\n{prettify_list(self.truths)}",
                    f"Claims:\n{prettify_list(self.claims)}",
                    f"Verdicts:\n{prettify_list(self.verdicts)}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )

            return self.score

    async def _a_generate_reason(self) -> str:
        if self.include_reason is False:
            return None

        contradictions = []
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "no":
                contradictions.append(verdict.reason)

        prompt: dict = MultimodalFaithfulnessTemplate.generate_reason(
            contradictions=contradictions,
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

    def _generate_reason(self) -> str:
        if self.include_reason is False:
            return None

        contradictions = []
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "no":
                contradictions.append(verdict.reason)

        prompt: dict = MultimodalFaithfulnessTemplate.generate_reason(
            contradictions=contradictions,
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

    async def _a_generate_verdicts(self) -> List[FaithfulnessVerdict]:
        if len(self.claims) == 0:
            return []

        verdicts: List[FaithfulnessVerdict] = []
        prompt = MultimodalFaithfulnessTemplate.generate_verdicts(
            claims=self.claims, retrieval_context="\n\n".join(self.truths)
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
                    FaithfulnessVerdict(**item) for item in data["verdicts"]
                ]
                return verdicts

    def _generate_verdicts(self) -> List[FaithfulnessVerdict]:
        if len(self.claims) == 0:
            return []

        verdicts: List[FaithfulnessVerdict] = []
        prompt = MultimodalFaithfulnessTemplate.generate_verdicts(
            claims=self.claims, retrieval_context="\n\n".join(self.truths)
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
                    FaithfulnessVerdict(**item) for item in data["verdicts"]
                ]
                return verdicts

    async def _a_generate_truths(
        self, retrieval_context: List[Union[str, MLLMImage]]
    ) -> List[str]:
        prompt = MultimodalFaithfulnessTemplate.generate_truths(
            excerpt=retrieval_context,
            extraction_limit=self.truths_extraction_limit,
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt, schema=Truths)
            self.evaluation_cost += cost
            return res.truths
        else:
            try:
                res: Truths = await self.model.a_generate(prompt, schema=Truths)
                return res.truths
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["truths"]

    def _generate_truths(
        self, retrieval_context: List[Union[str, MLLMImage]]
    ) -> List[str]:
        prompt = MultimodalFaithfulnessTemplate.generate_truths(
            excerpt=retrieval_context,
            extraction_limit=self.truths_extraction_limit,
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt, schema=Truths)
            self.evaluation_cost += cost
            return res.truths
        else:
            try:
                res: Truths = self.model.generate(prompt, schema=Truths)
                return res.truths
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["truths"]

    async def _a_generate_claims(
        self, actual_output: List[Union[str, MLLMImage]]
    ) -> List[str]:
        prompt = MultimodalFaithfulnessTemplate.generate_claims(
            excerpt=actual_output
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt, schema=Claims)
            self.evaluation_cost += cost
            return res.claims
        else:
            try:
                res: Claims = await self.model.a_generate(prompt, schema=Claims)
                return res.claims
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["claims"]

    def _generate_claims(
        self, actual_output: List[Union[str, MLLMImage]]
    ) -> List[str]:
        prompt = MultimodalFaithfulnessTemplate.generate_claims(
            excerpt=actual_output
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt, schema=Claims)
            self.evaluation_cost += cost
            return res.claims
        else:
            try:
                res: Claims = self.model.generate(prompt, schema=Claims)
                return res.claims
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["claims"]

    def _calculate_score(self) -> float:
        number_of_verdicts = len(self.verdicts)
        if number_of_verdicts == 0:
            return 1

        faithfulness_count = 0
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() != "no":
                faithfulness_count += 1

        score = faithfulness_count / number_of_verdicts
        return 0 if self.strict_mode and score < self.threshold else score

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
        return "Multimodal Faithfulness"
