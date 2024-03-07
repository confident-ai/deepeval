from typing import List, Optional, Union
import json
from pydantic import BaseModel, Field
import asyncio
from concurrent.futures import ThreadPoolExecutor

from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric
from deepeval.utils import trimAndLoadJson, get_or_create_event_loop
from deepeval.models import GPTModel, DeepEvalBaseLLM
from deepeval.metrics.faithfulness.template import FaithfulnessTemplate
from deepeval.progress_context import metrics_progress_context
from deepeval.telemetry import capture_metric_type


class FaithfulnessVerdict(BaseModel):
    verdict: str
    reason: str = Field(default=None)


class FaithfulnessMetric(BaseMetric):
    def __init__(
        self,
        threshold: float = 0.5,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        use_async: bool = True,
        strict_mode: bool = False,
    ):
        self.threshold = 1 if strict_mode else threshold
        if isinstance(model, DeepEvalBaseLLM):
            self.model = model
        else:
            self.model = GPTModel(model=model)
        self.evaluation_model = self.model.get_model_name()
        self.include_reason = include_reason
        self.use_async = use_async
        self.strict_mode = strict_mode

    def measure(self, test_case: LLMTestCase):
        if (
            test_case.input is None
            or test_case.actual_output is None
            or test_case.retrieval_context is None
        ):
            raise ValueError(
                "Input, actual output, and retrieval context cannot be None"
            )
        with metrics_progress_context(
            self.__name__, self.evaluation_model, self.strict_mode
        ):
            if self.use_async:
                loop = get_or_create_event_loop()
                self.truths, self.claims = loop.run_until_complete(
                    asyncio.gather(
                        self._a_generate_truths(test_case.retrieval_context),
                        self._a_generate_claims(test_case.actual_output),
                    )
                )
                self.verdicts = loop.run_until_complete(
                    self._a_generate_verdicts()
                )
                self.score = self._generate_score()
                self.reason = loop.run_until_complete(self._a_generate_reason())
            else:
                self.truths = self._generate_truths(test_case.retrieval_context)
                self.claims = self._generate_claims(test_case.actual_output)
                self.verdicts = self._generate_verdicts()
                self.score = self._generate_score()
                self.reason = self._generate_reason()

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

        if self.use_async:
            res = await self.model.a_generate(prompt)
        else:
            res = self.model(prompt)
        return res

    def _generate_reason(self) -> str:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self._a_generate_reason())

    def _generate_score(self) -> float:
        number_of_verdicts = len(self.verdicts)
        if number_of_verdicts == 0:
            return 0

        faithfulness_count = 0
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() != "no":
                faithfulness_count += 1

        score = faithfulness_count / number_of_verdicts

        return 0 if self.strict_mode and score < self.threshold else score

    async def _a_generate_verdicts(self) -> List[FaithfulnessVerdict]:
        verdicts: List[FaithfulnessVerdict] = []

        prompt = FaithfulnessTemplate.generate_verdicts(
            claims=self.claims, retrieval_context="\n\n".join(self.truths)
        )
        if self.use_async:
            res = await self.model.a_generate(prompt)
        else:
            res = self.model(prompt)
        data = trimAndLoadJson(res)
        verdicts = [FaithfulnessVerdict(**item) for item in data["verdicts"]]

        return verdicts

    def _generate_verdicts(self) -> List[FaithfulnessVerdict]:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self._a_generate_verdicts())

    async def _a_generate_truths(self, retrieval_context: str) -> List[str]:
        print("generating truths")
        prompt = FaithfulnessTemplate.generate_claims(
            text="\n\n".join(retrieval_context)
        )
        if self.use_async:
            res = await self.model.a_generate(prompt)
        else:
            res = self.model(prompt)
        data = trimAndLoadJson(res)

        return data["claims"]

    def _generate_truths(self, retrieval_context: str) -> List[str]:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self._a_generate_truths(retrieval_context)
        )

    async def _a_generate_claims(self, actual_output: str) -> List[str]:
        print("generating claims")
        prompt = FaithfulnessTemplate.generate_claims(text=actual_output)
        if self.use_async:
            res = await self.model.a_generate(prompt)
        else:
            res = self.model(prompt)
        data = trimAndLoadJson(res)
        return data["claims"]

    def _generate_claims(self, actual_output: str) -> List[str]:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self._a_generate_claims(actual_output))

    def is_successful(self) -> bool:
        self.success = self.score >= self.threshold
        return self.success

    @property
    def __name__(self):
        return "Faithfulness"
