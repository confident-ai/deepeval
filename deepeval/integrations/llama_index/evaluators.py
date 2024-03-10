from typing import Optional, Sequence, Any
from llama_index.core.evaluation.base import BaseEvaluator, EvaluationResult
import asyncio

from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    SummarizationMetric,
    ContextualRelevancyMetric,
    BiasMetric,
    ToxicityMetric,
)
from deepeval.integrations.llama_index.utils import conform_contexts_type


class AnswerRelevancyEvaluator(BaseEvaluator):
    def __init__(
        self,
        threshold: float = 0.5,
        include_reason: bool = True,
        model: Optional[str] = None,
    ):
        self.threshold = threshold
        self.include_reason = include_reason
        self.model = model

    def _get_prompts(self):
        pass

    def _update_prompts(self):
        pass

    async def aevaluate(
        self,
        query: Optional[str] = None,
        response: Optional[str] = None,
        contexts: Optional[Sequence[str]] = None,
        sleep_time_in_seconds: int = 0,
        **kwargs: Any,
    ) -> EvaluationResult:
        del kwargs  # Unused

        await asyncio.sleep(sleep_time_in_seconds)

        if query is None or response is None or contexts is None:
            raise ValueError("Query, response, and contexts must be provided")

        test_case = LLMTestCase(
            input=query,
            actual_output=response,
            retrieval_context=conform_contexts_type(contexts),
        )
        metric = AnswerRelevancyMetric(
            threshold=self.threshold,
            include_reason=self.include_reason,
            model=self.model,
        )
        await metric.a_measure(test_case)
        return EvaluationResult(
            query=query,
            response=response,
            passing=metric.is_successful(),
            score=metric.score,
            feedback=metric.reason,
        )


class FaithfulnessEvaluator(BaseEvaluator):
    def __init__(
        self,
        threshold: float = 0.5,
        include_reason: bool = True,
        model: Optional[str] = None,
    ):
        self.threshold = threshold
        self.include_reason = include_reason
        self.model = model

    def _get_prompts(self):
        pass

    def _update_prompts(self):
        pass

    async def aevaluate(
        self,
        query: Optional[str] = None,
        response: Optional[str] = None,
        contexts: Optional[Sequence[str]] = None,
        sleep_time_in_seconds: int = 0,
        **kwargs: Any,
    ) -> EvaluationResult:
        del kwargs  # Unused

        await asyncio.sleep(sleep_time_in_seconds)

        if query is None or response is None or contexts is None:
            raise ValueError("Query, response, and contexts must be provided")

        test_case = LLMTestCase(
            input=query,
            actual_output=response,
            retrieval_context=conform_contexts_type(contexts),
        )
        metric = FaithfulnessMetric(
            threshold=self.threshold,
            include_reason=self.include_reason,
            model=self.model,
        )
        await metric.a_measure(test_case)
        return EvaluationResult(
            query=query,
            response=response,
            passing=metric.is_successful(),
            score=metric.score,
            feedback=metric.reason,
        )


class ContextualRelevancyEvaluator(BaseEvaluator):
    def __init__(
        self,
        threshold: float = 0.5,
        include_reason: bool = True,
        model: Optional[str] = None,
    ):
        self.threshold = threshold
        self.include_reason = include_reason
        self.model = model

    def _get_prompts(self):
        pass

    def _update_prompts(self):
        pass

    async def aevaluate(
        self,
        query: Optional[str] = None,
        response: Optional[str] = None,
        contexts: Optional[Sequence[str]] = None,
        sleep_time_in_seconds: int = 0,
        **kwargs: Any,
    ) -> EvaluationResult:
        del kwargs  # Unused

        await asyncio.sleep(sleep_time_in_seconds)

        if query is None or response is None or contexts is None:
            raise ValueError("Query, response, and contexts must be provided")

        test_case = LLMTestCase(
            input=query,
            actual_output=response,
            retrieval_context=conform_contexts_type(contexts),
        )
        metric = ContextualRelevancyMetric(
            threshold=self.threshold,
            include_reason=self.include_reason,
            model=self.model,
        )
        await metric.a_measure(test_case)
        return EvaluationResult(
            query=query,
            response=response,
            passing=metric.is_successful(),
            score=metric.score,
            feedback=metric.reason,
        )


class SummarizationEvaluator(BaseEvaluator):
    def __init__(
        self,
        threshold: float = 0.5,
        include_reason: bool = True,
        model: Optional[str] = None,
    ):
        self.threshold = threshold
        self.include_reason = include_reason
        self.model = model

    def _get_prompts(self):
        pass

    def _update_prompts(self):
        pass

    async def aevaluate(
        self,
        query: Optional[str] = None,
        response: Optional[str] = None,
        contexts: Optional[Sequence[str]] = None,
        sleep_time_in_seconds: int = 0,
        **kwargs: Any,
    ) -> EvaluationResult:
        del kwargs  # Unused
        del contexts  # Unused

        await asyncio.sleep(sleep_time_in_seconds)

        if query is None or response is None:
            raise ValueError("Query and response must be provided")

        test_case = LLMTestCase(input=query, actual_output=response)
        metric = SummarizationMetric(
            threshold=self.threshold,
            model=self.model,
            include_reason=self.include_reason,
        )
        await metric.a_measure(test_case)
        return EvaluationResult(
            query=query,
            response=response,
            passing=metric.is_successful(),
            score=metric.score,
            feedback=metric.reason,
        )


class BiasEvaluator(BaseEvaluator):
    def __init__(
        self,
        threshold: float = 0.5,
        include_reason: bool = True,
        model: Optional[str] = None,
    ):
        self.threshold = threshold
        self.include_reason = include_reason
        self.model = model

    def _get_prompts(self):
        pass

    def _update_prompts(self):
        pass

    async def aevaluate(
        self,
        query: Optional[str] = None,
        response: Optional[str] = None,
        contexts: Optional[Sequence[str]] = None,
        sleep_time_in_seconds: int = 0,
        **kwargs: Any,
    ) -> EvaluationResult:
        del kwargs  # Unused
        del contexts  # Unused

        await asyncio.sleep(sleep_time_in_seconds)

        if query is None or response is None:
            raise ValueError("Query and response must be provided")

        test_case = LLMTestCase(
            input=query,
            actual_output=response,
        )
        metric = BiasMetric(
            threshold=self.threshold,
            model=self.model,
            include_reason=self.include_reason,
        )
        await metric.a_measure(test_case)
        return EvaluationResult(
            query=query,
            response=response,
            passing=metric.is_successful(),
            score=metric.score,
            feedback=metric.reason,
        )


class ToxicityEvaluator(BaseEvaluator):
    def __init__(
        self,
        threshold: float = 0.5,
        include_reason: bool = True,
        model: Optional[str] = None,
    ):
        self.threshold = threshold
        self.include_reason = include_reason
        self.model = model

    def _get_prompts(self):
        pass

    def _update_prompts(self):
        pass

    async def aevaluate(
        self,
        query: Optional[str] = None,
        response: Optional[str] = None,
        contexts: Optional[Sequence[str]] = None,
        sleep_time_in_seconds: int = 0,
        **kwargs: Any,
    ) -> EvaluationResult:
        del kwargs  # Unused
        del contexts  # Unused

        await asyncio.sleep(sleep_time_in_seconds)

        if query is None or response is None:
            raise ValueError("Query and response must be provided")

        test_case = LLMTestCase(
            input=query,
            actual_output=response,
        )
        metric = ToxicityMetric(
            threshold=self.threshold,
            model=self.model,
            include_reason=self.include_reason,
        )
        await metric.a_measure(test_case)
        return EvaluationResult(
            query=query,
            response=response,
            passing=metric.is_successful(),
            score=metric.score,
            feedback=metric.reason,
        )
