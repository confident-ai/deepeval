from typing import Optional, List, Tuple, Union

from deepeval.metrics import BaseMetric
from deepeval.test_case import (
    LLMTestCase,
    LLMTestCaseParams,
    ConversationalTestCase,
)
from deepeval.metrics.red_teaming_metrics.pii.template import PIITemplate
from deepeval.utils import get_or_create_event_loop, prettify_list
from deepeval.metrics.utils import (
    construct_verbose_logs,
    check_conversational_test_case_params,
    trimAndLoadJson,
    check_llm_test_case_params,
    initialize_model,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.red_teaming_metrics.pii.schema import *

required_params: List[LLMTestCaseParams] = [
    LLMTestCaseParams.INPUT,
    LLMTestCaseParams.ACTUAL_OUTPUT,
]


class PIIGrader(BaseMetric):
    def __init__(
        self,
        purpose: str,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        async_mode: bool = True,
        verbose_mode: bool = False,
    ):
        self.purpose = purpose
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.async_mode = async_mode
        self.verbose_mode = verbose_mode
        try:
            self.system_prompt = model.get_system_prompt()
        except:
            self.system_prompt = ""

    def measure(
        self, test_case: Union[LLMTestCase, ConversationalTestCase]
    ) -> float:
        if isinstance(test_case, ConversationalTestCase):
            test_case = check_conversational_test_case_params(test_case, self)
        check_llm_test_case_params(test_case, required_params, self)

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(self, _show_indicator=False):
            if self.async_mode:
                loop = get_or_create_event_loop()
                loop.run_until_complete(
                    self.a_measure(test_case, _show_indicator=False)
                )
            else:
                self.purpose = self.generate_purpose()
                self.entities = self.generate_entities()
                score, reason = self.evaluate(test_case)
                self.reason = reason
                self.score = score
                self.success = self.score == 1
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Purpose: {self.purpose}\nEntities: {self.entities}\nScore: {self.score}\nReason: {self.reason}",
                    ],
                )
                return self.score

    async def a_measure(
        self,
        test_case: Union[LLMTestCase, ConversationalTestCase],
        _show_indicator: bool = False,
    ) -> float:
        if isinstance(test_case, ConversationalTestCase):
            test_case = check_conversational_test_case_params(test_case, self)
        check_llm_test_case_params(test_case, required_params, self)

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
        ):
            self.purpose = await self.a_generate_purpose()
            self.entities = await self.a_generate_entities()
            score, reason = await self._a_evaluate(test_case)
            self.reason = reason
            self.score = score
            self.success = self.score == 1
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Purpose: {self.purpose}\nEntities: {self.entities}\nScore: {self.score}\nReason: {self.reason}",
                ],
            )
            return self.score

    async def a_generate_purpose(self) -> str:
        if self.purpose is not None:
            return self.purpose
        prompt = PIITemplate.extract_purpose(self.system_prompt)
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return data["purpose"]
        else:
            try:
                res: Purpose = await self.model.a_generate(
                    prompt, schema=Purpose
                )
                return res.purpose
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["purpose"]

    def generate_purpose(self) -> str:
        if self.purpose is not None:
            return self.purpose
        prompt = PIITemplate.extract_purpose(self.system_prompt)
        if self.using_native_model:
            res, cost = self.model.generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return data["purpose"]
        else:
            try:
                res: Purpose = self.model.generate(prompt, schema=Purpose)
                return res.purpose
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["purpose"]

    async def a_generate_entities(self) -> List[str]:
        prompt = PIITemplate.extract_entities(self.system_prompt)
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return data["entities"]
        else:
            try:
                res: Entities = await self.model.a_generate(
                    prompt, schema=Entities
                )
                return res.entities
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["entities"]

    def generate_entities(self) -> List[str]:
        prompt = PIITemplate.extract_entities(self.system_prompt)
        if self.using_native_model:
            res, cost = self.model.generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return data["entities"]
        else:
            try:
                res: Entities = self.model.generate(prompt, schema=Entities)
                return res.entities
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["entities"]

    async def _a_evaluate(
        self, test_case: LLMTestCase
    ) -> Tuple[Union[int, float], str]:
        prompt = PIITemplate.generate_evaluation_results(
            input=test_case.input,
            actual_output=test_case.actual_output,
            entities=self.entities,
            purpose=self.purpose,
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return data["score"], data["reason"]
        else:
            try:
                res: ReasonScore = await self.model.a_generate(
                    prompt, schema=ReasonScore
                )
                return res.score, res.reason
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["score"], data["reason"]

    def evaluate(self, test_case: LLMTestCase) -> Tuple[Union[int, float], str]:
        prompt = PIITemplate.generate_evaluation_results(
            input=test_case.input,
            actual_output=test_case.actual_output,
            entities=self.entities,
            purpose=self.purpose,
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return data["score"], data["reason"]
        else:
            try:
                res: ReasonScore = self.model.generate(
                    prompt, schema=ReasonScore
                )
                return res.score, res.reason
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["score"], data["reason"]

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            try:
                self.score == 1
            except:
                self.success = False
        return self.success

    @property
    def __name__(self):
        return f"PII (Red Teaming)"
