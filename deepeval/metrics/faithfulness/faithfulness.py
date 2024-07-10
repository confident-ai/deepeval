from typing import List, Optional, Union
from pydantic import BaseModel, Field
import asyncio
import inspect

from deepeval.test_case import (
    LLMTestCase,
    LLMTestCaseParams,
    ConversationalTestCase,
)
from deepeval.metrics import BaseMetric
from deepeval.utils import get_or_create_event_loop, prettify_list
from deepeval.metrics.utils import (
    print_intermediate_steps,
    validate_conversational_test_case,
    trimAndLoadJson,
    check_llm_test_case_params,
    initialize_model,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.faithfulness.template import FaithfulnessTemplate
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.faithfulness.models import *

required_params: List[LLMTestCaseParams] = [
    LLMTestCaseParams.INPUT,
    LLMTestCaseParams.ACTUAL_OUTPUT,
    LLMTestCaseParams.RETRIEVAL_CONTEXT,
]

class FaithfulnessMetric(BaseMetric):
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
                self.truths = self._generate_truths(test_case.retrieval_context)
                self.claims = self._generate_claims(test_case.actual_output)
                self.verdicts = self._generate_verdicts()
                self.score = self._calculate_score()
                self.reason = self._generate_reason()
                self.success = self.score >= self.threshold
                if self.verbose_mode:
                    print_intermediate_steps(
                        self.__name__,
                        steps=[
                            f"Truths:\n{prettify_list(self.truths)}\n",
                            f"Claims:\n{prettify_list(self.claims)}\n",
                            f"Verdicts:\n{prettify_list(self.verdicts)}\n",
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
            if self.verbose_mode:
                print_intermediate_steps(
                    self.__name__,
                    steps=[
                        f"Truths:\n{prettify_list(self.truths)}\n",
                        f"Claims:\n{prettify_list(self.claims)}\n",
                        f"Verdicts:\n{prettify_list(self.verdicts)}\n",
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

        prompt: dict = FaithfulnessTemplate.generate_reason(
            contradictions=contradictions,
            score=format(self.score, ".2f"),
        )

        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return data["reason"]
        elif 'pydantic_model' in inspect.signature(self.model.a_generate).parameters:
            res: Reason = await self.model.a_generate(prompt, Reason)
            return res.reason
        else:
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

        prompt: dict = FaithfulnessTemplate.generate_reason(
            contradictions=contradictions,
            score=format(self.score, ".2f"),
        )

        if self.using_native_model:
            res, cost = self.model.generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return data["reason"]
        elif 'pydantic_model' in inspect.signature(self.model.generate).parameters:
            res: Reason = self.model.generate(prompt, Reason)
            return res.reason
        else:
            res = self.model.generate(prompt)
            data = trimAndLoadJson(res, self)
            return data["reason"]
            
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
            data = trimAndLoadJson(res, self)
            verdicts = [FaithfulnessVerdict(**item) for item in data["verdicts"]]
            return verdicts
        elif 'pydantic_model' in inspect.signature(self.model.a_generate).parameters:
            res: Verdicts = await self.model.a_generate(prompt, Verdicts)
            verdicts = [item for item in res.verdicts]
            return verdicts
        else:
            res = await self.model.a_generate(prompt)
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
            data = trimAndLoadJson(res, self)
            verdicts = [FaithfulnessVerdict(**item) for item in data["verdicts"]]
            return verdicts
        elif 'pydantic_model' in inspect.signature(self.model.generate).parameters:
            res: Verdicts = self.model.generate(prompt, Verdicts)
            verdicts = [item for item in res.verdicts]
            return verdicts
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
            data = trimAndLoadJson(res, self)
            return data["truths"]
        elif 'pydantic_model' in inspect.signature(self.model.a_generate).parameters:
            res: Truth = await self.model.a_generate(prompt, Truth)
            return res.truths
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
            data = trimAndLoadJson(res, self)
            return data["truths"]
        elif 'pydantic_model' in inspect.signature(self.model.generate).parameters:
            res: Truth = self.model.generate(prompt, Truth)
            return res.truths
        else:
            res = self.model.generate(prompt)
            data = trimAndLoadJson(res, self)
            return data["truths"]

    async def _a_generate_claims(self, actual_output: str) -> List[str]:
        prompt = FaithfulnessTemplate.generate_claims(text=actual_output)
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return data["claims"]
        elif 'pydantic_model' in inspect.signature(self.model.a_generate).parameters:
            res: Claim = await self.model.a_generate(prompt, Claim)
            return res.claims
        else:
            res = await self.model.a_generate(prompt)
            data = trimAndLoadJson(res, self)
            return data["claims"]

    def _generate_claims(self, actual_output: str) -> List[str]:
        prompt = FaithfulnessTemplate.generate_claims(text=actual_output)
        if self.using_native_model:
            res, cost = self.model.generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return data["claims"]
        elif 'pydantic_model' in inspect.signature(self.model.generate).parameters:
            res: Claim = self.model.generate(prompt, Claim)
            return res.claims
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
