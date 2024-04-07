from typing import Optional, List, Union
from pydantic import BaseModel

from deepeval.utils import get_or_create_event_loop
from deepeval.metrics.utils import trimAndLoadJson, check_test_case_params
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import BaseMetric
from deepeval.models import GPTModel, DeepEvalBaseLLM
from deepeval.metrics.contextual_precision.template import (
    ContextualPrecisionTemplate,
)
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.telemetry import capture_metric_type


required_params: List[LLMTestCaseParams] = [
    LLMTestCaseParams.INPUT,
    LLMTestCaseParams.ACTUAL_OUTPUT,
    LLMTestCaseParams.RETRIEVAL_CONTEXT,
    LLMTestCaseParams.EXPECTED_OUTPUT,
]


class ContextualPrecisionVerdict(BaseModel):
    verdict: str
    reason: str


class ContextualPrecisionMetric(BaseMetric):
    def __init__(
        self,
        threshold: float = 0.5,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.include_reason = include_reason
        if isinstance(model, DeepEvalBaseLLM):
            self.using_native_model = False
            self.model = model
        else:
            self.using_native_model = True
            self.model = GPTModel(model=model)
        self.evaluation_model = self.model.get_model_name()
        self.async_mode = async_mode
        self.strict_mode = strict_mode

    def measure(self, test_case: LLMTestCase) -> float:
        check_test_case_params(test_case, required_params, self)
        self.evaluation_cost = 0 if self.using_native_model else None

        with metric_progress_indicator(self):
            if self.async_mode:
                loop = get_or_create_event_loop()
                loop.run_until_complete(
                    self.a_measure(test_case, _show_indicator=False)
                )
            else:
                self.verdicts: List[ContextualPrecisionVerdict] = (
                    self._generate_verdicts(
                        test_case.input,
                        test_case.expected_output,
                        test_case.retrieval_context,
                    )
                )
                self.score = self._calculate_score()
                self.reason = self._generate_reason(test_case.input)
                self.success = self.score >= self.threshold
                capture_metric_type(self.__name__)
                return self.score

    async def a_measure(
        self, test_case: LLMTestCase, _show_indicator: bool = True
    ) -> float:
        check_test_case_params(test_case, required_params, self)
        self.evaluation_cost = 0 if self.using_native_model else None

        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
        ):
            self.verdicts: List[ContextualPrecisionVerdict] = (
                await self._a_generate_verdicts(
                    test_case.input,
                    test_case.expected_output,
                    test_case.retrieval_context,
                )
            )
            self.score = self._calculate_score()
            self.reason = await self._a_generate_reason(test_case.input)
            self.success = self.score >= self.threshold
            capture_metric_type(self.__name__)
            return self.score

    async def _a_generate_reason(self, input: str):
        if self.include_reason is False:
            return None

        retrieval_contexts_verdicts = [
            {"verdict": verdict.verdict, "reasons": verdict.reason}
            for verdict in self.verdicts
        ]
        prompt = ContextualPrecisionTemplate.generate_reason(
            input=input,
            verdicts=retrieval_contexts_verdicts,
            score=format(self.score, ".2f"),
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt)
            self.evaluation_cost += cost
        else:
            res = await self.model.a_generate(prompt)
        return res

    def _generate_reason(self, input: str):
        if self.include_reason is False:
            return None

        retrieval_contexts_verdicts = [
            {"verdict": verdict.verdict, "reasons": verdict.reason}
            for verdict in self.verdicts
        ]
        prompt = ContextualPrecisionTemplate.generate_reason(
            input=input,
            verdicts=retrieval_contexts_verdicts,
            score=format(self.score, ".2f"),
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt)
            self.evaluation_cost += cost
        else:
            res = self.model.generate(prompt)
        return res

    async def _a_generate_verdicts(
        self, input: str, expected_output: str, retrieval_context: List[str]
    ) -> List[ContextualPrecisionVerdict]:
        prompt = ContextualPrecisionTemplate.generate_verdicts(
            input=input,
            expected_output=expected_output,
            retrieval_context=retrieval_context,
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt)
            self.evaluation_cost += cost
        else:
            res = await self.model.a_generate(prompt)
        data = trimAndLoadJson(res, self)
        verdicts = [
            ContextualPrecisionVerdict(**item) for item in data["verdicts"]
        ]
        return verdicts

    def _generate_verdicts(
        self, input: str, expected_output: str, retrieval_context: List[str]
    ) -> List[ContextualPrecisionVerdict]:
        prompt = ContextualPrecisionTemplate.generate_verdicts(
            input=input,
            expected_output=expected_output,
            retrieval_context=retrieval_context,
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt)
            self.evaluation_cost += cost
        else:
            res = self.model.generate(prompt)
        data = trimAndLoadJson(res, self)
        verdicts = [
            ContextualPrecisionVerdict(**item) for item in data["verdicts"]
        ]
        return verdicts

    def _calculate_score(self):
        number_of_verdicts = len(self.verdicts)
        if number_of_verdicts == 0:
            return 0

        # Convert verdicts to a binary list where 'yes' is 1 and others are 0
        node_verdicts = [
            1 if v.verdict.strip().lower() == "yes" else 0
            for v in self.verdicts
        ]

        sum_weighted_precision_at_k = 0.0
        relevant_nodes_count = 0
        for k, is_relevant in enumerate(node_verdicts, start=1):
            # If the item is relevant, update the counter and add the weighted precision at k to the sum
            if is_relevant:
                relevant_nodes_count += 1
                precision_at_k = relevant_nodes_count / k
                sum_weighted_precision_at_k += precision_at_k * is_relevant

        if relevant_nodes_count == 0:
            return 0
        # Calculate weighted cumulative precision
        score = sum_weighted_precision_at_k / relevant_nodes_count
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
        return "Contextual Precision"
