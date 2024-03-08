import asyncio
from typing import Optional, List, Union
from pydantic import BaseModel, Field
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

from deepeval.utils import trimAndLoadJson, get_or_create_event_loop
from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric
from deepeval.models import GPTModel, DeepEvalBaseLLM
from deepeval.metrics.contextual_relevancy.template import (
    ContextualRelevancyTemplate,
)
from deepeval.progress_context import metrics_progress_context
from deepeval.telemetry import capture_metric_type


class ContextualRelevancyVerdict(BaseModel):
    verdict: str
    reason: str = Field(default=None)


class ContextualRelevancyMetric(BaseMetric):
    def __init__(
        self,
        threshold: float = 0.5,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        asynchronous: bool = True,
        strict_mode: bool = False,
    ):
        self.threshold = 1 if strict_mode else threshold
        if isinstance(model, DeepEvalBaseLLM):
            self.model = model
        else:
            self.model = GPTModel(model=model)
        self.evaluation_model = self.model.get_model_name()
        self.include_reason = include_reason
        self.asynchronous = asynchronous
        self.strict_mode = strict_mode

    def measure(
        self, test_case: LLMTestCase, _asynchronous: Optional[bool] = None
    ) -> float:
        if (
            test_case.input is None
            or test_case.actual_output is None
            or test_case.retrieval_context is None
        ):
            raise ValueError(
                "Input, actual output, or retrieval context cannot be None"
            )
        asynchronous = (
            _asynchronous if _asynchronous is not None else self.asynchronous
        )
        with metrics_progress_context(
            self.__name__,
            self.evaluation_model,
            self.strict_mode,
            asynchronous,
        ):
            if asynchronous:
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
                capture_metric_type(self.__name__)
                return self.score

    async def a_measure(
        self, test_case: LLMTestCase, _show_indicator: bool = True
    ) -> float:
        with metrics_progress_context(
            self.__name__,
            self.evaluation_model,
            self.strict_mode,
            True,
            _show_indicator,
        ):
            print("a contextual relevancy")
            self.verdicts: List[ContextualRelevancyVerdict] = (
                await self._a_generate_verdicts(
                    test_case.input, test_case.retrieval_context
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

        irrelevancies = []
        for verdict in self.verdicts:
            if verdict.verdict.lower() == "no":
                irrelevancies.append(verdict.reason)

        prompt: dict = ContextualRelevancyTemplate.generate_reason(
            input=input,
            irrelevancies=irrelevancies,
            score=format(self.score, ".2f"),
        )
        res = await self.model.a_generate(prompt)
        return res

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
        res = self.model.generate(prompt)
        return res

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

    async def _a_generate_verdicts(
        self, text: str, retrieval_context: List[str]
    ) -> ContextualRelevancyVerdict:
        tasks = []
        for context in retrieval_context:
            prompt = ContextualRelevancyTemplate.generate_verdict(
                text=text, context=context
            )
            task = asyncio.create_task(self.model.a_generate(prompt))
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        verdicts = []
        for res in results:
            data = trimAndLoadJson(res)
            verdict = ContextualRelevancyVerdict(**data)
            verdicts.append(verdict)

        return verdicts

    def _generate_verdicts(
        self, text: str, retrieval_context: List[str]
    ) -> List[ContextualRelevancyVerdict]:
        verdicts: List[ContextualRelevancyVerdict] = []
        for context in retrieval_context:
            prompt = ContextualRelevancyTemplate.generate_verdict(
                text=text, context=context
            )
            res = self.model.generate(prompt)
            data = trimAndLoadJson(res)
            verdict = ContextualRelevancyVerdict(**data)
            verdicts.append(verdict)

        return verdicts

    def is_successful(self) -> bool:
        self.success = self.score >= self.threshold
        return self.success

    @property
    def __name__(self):
        return "Contextual Relevancy"
