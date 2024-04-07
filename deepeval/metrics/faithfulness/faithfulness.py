from typing import List, Optional, Union
from pydantic import BaseModel, Field
import asyncio

from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import BaseMetric
from deepeval.utils import get_or_create_event_loop
from deepeval.metrics.utils import trimAndLoadJson, check_test_case_params
from deepeval.models import GPTModel, DeepEvalBaseLLM
from deepeval.metrics.faithfulness.template import FaithfulnessTemplate
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.telemetry import capture_metric_type

required_params: List[LLMTestCaseParams] = [
    LLMTestCaseParams.INPUT,
    LLMTestCaseParams.ACTUAL_OUTPUT,
    LLMTestCaseParams.RETRIEVAL_CONTEXT,
]


class FaithfulnessVerdict(BaseModel):
    verdict: str
    reason: str = Field(default=None)


class FaithfulnessMetric(BaseMetric):
    def __init__(
        self,
        threshold: float = 0.5,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
    ):
        self.threshold = 1 if strict_mode else threshold
        if isinstance(model, DeepEvalBaseLLM):
            self.using_native_model = False
            self.model = model
        else:
            self.using_native_model = True
            self.model = GPTModel(model=model)
        self.evaluation_model = self.model.get_model_name()
        self.include_reason = include_reason
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
                self.truths = self._generate_truths(test_case.retrieval_context)
                self.claims = self._generate_claims(test_case.actual_output)
                self.verdicts = self._generate_verdicts()
                self.score = self._calculate_score()
                self.reason = self._generate_reason()
                self.success = self.score >= self.threshold
                capture_metric_type(self.__name__)
                return self.score

    async def a_measure(
        self, test_case: LLMTestCase, _show_indicator: bool = True
    ) -> float:
        check_test_case_params(test_case, required_params, self)
        self.evaluation_cost = 0 if self.using_native_model else None

        with metric_progress_indicator(
            self, async_mode=True, _show_indicator=_show_indicator
        ):
            self.truths, self.claims = await asyncio.gather(
                self._a_generate_truths(test_case.retrieval_context),
                self._a_generate_claims(test_case.actual_output),
            )
            self.verdicts = await self._a_generate_verdicts()
            self.score = self._calculate_score()
            self.reason = await self._a_generate_reason()
            self.success = self.score >= self.threshold
            capture_metric_type(self.__name__)
            return self.score

    async def _a_generate_reason(self) -> str:
        if self.include_reason is False:
            return None

        contradictions = []
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "no":
                contradictions.append(verdict.reason)

        prompt: dict = FaithfulnessTemplate.generate_reason(
            contradictions=contradictions,
            score=format(self.score, ".2f"),
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt)
            self.evaluation_cost += cost
        else:
            res = await self.model.a_generate(prompt)
        return res

    def _generate_reason(self) -> str:
        if self.include_reason is False:
            return None

        contradictions = []
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "no":
                contradictions.append(verdict.reason)

        prompt: dict = FaithfulnessTemplate.generate_reason(
            contradictions=contradictions,
            score=format(self.score, ".2f"),
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt)
            self.evaluation_cost += cost
        else:
            res = self.model.generate(prompt)

        return res

    async def _a_generate_verdicts(self) -> List[FaithfulnessVerdict]:
        if len(self.claims) == 0:
            return []

        verdicts: List[FaithfulnessVerdict] = []
        prompt = FaithfulnessTemplate.generate_verdicts(
            claims=self.claims, retrieval_context="\n\n".join(self.truths)
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt)
            self.evaluation_cost += cost
        else:
            res = self.model.generate(prompt)

        data = trimAndLoadJson(res, self)
        verdicts = [FaithfulnessVerdict(**item) for item in data["verdicts"]]
        return verdicts

    def _generate_verdicts(self) -> List[FaithfulnessVerdict]:
        if len(self.claims) == 0:
            return []

        verdicts: List[FaithfulnessVerdict] = []
        prompt = FaithfulnessTemplate.generate_verdicts(
            claims=self.claims, retrieval_context="\n\n".join(self.truths)
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt)
            self.evaluation_cost += cost
        else:
            res = self.model.generate(prompt)

        data = trimAndLoadJson(res, self)
        verdicts = [FaithfulnessVerdict(**item) for item in data["verdicts"]]
        return verdicts

    async def _a_generate_truths(self, retrieval_context: str) -> List[str]:
        prompt = FaithfulnessTemplate.generate_truths(
            text="\n\n".join(retrieval_context)
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt)
            self.evaluation_cost += cost
        else:
            res = await self.model.a_generate(prompt)
        data = trimAndLoadJson(res, self)
        return data["truths"]

    def _generate_truths(self, retrieval_context: str) -> List[str]:
        prompt = FaithfulnessTemplate.generate_truths(
            text="\n\n".join(retrieval_context)
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt)
            self.evaluation_cost += cost
        else:
            res = self.model.generate(prompt)

        data = trimAndLoadJson(res, self)
        return data["truths"]

    async def _a_generate_claims(self, actual_output: str) -> List[str]:
        prompt = FaithfulnessTemplate.generate_claims(text=actual_output)
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt)
            self.evaluation_cost += cost
        else:
            res = await self.model.a_generate(prompt)

        data = trimAndLoadJson(res, self)
        return data["claims"]

    def _generate_claims(self, actual_output: str) -> List[str]:
        prompt = FaithfulnessTemplate.generate_claims(text=actual_output)
        if self.using_native_model:
            res, cost = self.model.generate(prompt)
            self.evaluation_cost += cost
        else:
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
        return "Faithfulness"
