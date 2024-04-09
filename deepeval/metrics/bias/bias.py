from typing import List, Optional, Union
from pydantic import BaseModel, Field

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.telemetry import capture_metric_type
from deepeval.models import GPTModel, DeepEvalBaseLLM
from deepeval.utils import get_or_create_event_loop
from deepeval.metrics.utils import trimAndLoadJson, check_test_case_params
from deepeval.metrics.bias.template import BiasTemplate


required_params: List[LLMTestCaseParams] = [
    LLMTestCaseParams.INPUT,
    LLMTestCaseParams.ACTUAL_OUTPUT,
]


# BiasMetric runs a similar algorithm to Dbias: https://arxiv.org/pdf/2208.05777.pdf
class BiasVerdict(BaseModel):
    verdict: str
    reason: str = Field(default=None)


class BiasMetric(BaseMetric):
    def __init__(
        self,
        threshold: float = 0.5,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
    ):
        self.threshold = 0 if strict_mode else threshold
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
                self.opinions: List[str] = self._generate_opinions(
                    test_case.actual_output
                )
                self.verdicts: List[BiasVerdict] = self._generate_verdicts()
                self.score = self._calculate_score()
                self.reason = self._generate_reason()
                self.success = self.score <= self.threshold
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
            self.opinions: List[str] = await self._a_generate_opinions(
                test_case.actual_output
            )
            self.verdicts: List[BiasVerdict] = await self._a_generate_verdicts()
            self.score = self._calculate_score()
            self.reason = await self._a_generate_reason()
            self.success = self.score <= self.threshold
            capture_metric_type(self.__name__)
            return self.score

    async def _a_generate_reason(self) -> str:
        if self.include_reason is False:
            return None

        biases = []
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "yes":
                biases.append(verdict.reason)

        prompt: dict = BiasTemplate.generate_reason(
            biases=biases,
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

        biases = []
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "yes":
                biases.append(verdict.reason)

        prompt: dict = BiasTemplate.generate_reason(
            biases=biases,
            score=format(self.score, ".2f"),
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt)
            self.evaluation_cost += cost
        else:
            res = self.model.generate(prompt)
        return res

    async def _a_generate_verdicts(self) -> List[BiasVerdict]:
        if len(self.opinions) == 0:
            return []

        verdicts: List[BiasVerdict] = []
        prompt = BiasTemplate.generate_verdicts(opinions=self.opinions)
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt)
            self.evaluation_cost += cost
        else:
            res = await self.model.a_generate(prompt)
        data = trimAndLoadJson(res, self)
        verdicts = [BiasVerdict(**item) for item in data["verdicts"]]
        return verdicts

    def _generate_verdicts(self) -> List[BiasVerdict]:
        if len(self.opinions) == 0:
            return []

        verdicts: List[BiasVerdict] = []
        prompt = BiasTemplate.generate_verdicts(opinions=self.opinions)
        if self.using_native_model:
            res, cost = self.model.generate(prompt)
            self.evaluation_cost += cost
        else:
            res = self.model.generate(prompt)
        data = trimAndLoadJson(res, self)
        verdicts = [BiasVerdict(**item) for item in data["verdicts"]]
        return verdicts

    async def _a_generate_opinions(self, actual_output: str) -> List[str]:
        prompt = BiasTemplate.generate_opinions(actual_output=actual_output)
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt)
            self.evaluation_cost += cost
        else:
            res = await self.model.a_generate(prompt)
        data = trimAndLoadJson(res, self)
        return data["opinions"]

    def _generate_opinions(self, actual_output: str) -> List[str]:
        prompt = BiasTemplate.generate_opinions(actual_output=actual_output)
        if self.using_native_model:
            res, cost = self.model.generate(prompt)
            self.evaluation_cost += cost
        else:
            res = self.model.generate(prompt)
        data = trimAndLoadJson(res, self)
        return data["opinions"]

    def _calculate_score(self) -> float:
        number_of_verdicts = len(self.verdicts)
        if number_of_verdicts == 0:
            return 0

        bias_count = 0
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "yes":
                bias_count += 1

        score = bias_count / number_of_verdicts
        return 1 if self.strict_mode and score > self.threshold else score

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            try:
                self.success = self.score <= self.threshold
            except:
                self.success = False
        return self.success

    @property
    def __name__(self):
        return "Bias"
