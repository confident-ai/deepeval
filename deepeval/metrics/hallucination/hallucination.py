import asyncio
from typing import Optional, Union, List
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel, Field

from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric
from deepeval.utils import trimAndLoadJson, get_or_create_event_loop
from deepeval.metrics.hallucination.template import HallucinationTemplate
from deepeval.models import GPTModel, DeepEvalBaseLLM
from deepeval.progress_context import metrics_progress_context
from deepeval.telemetry import capture_metric_type


class HallucinationVerdict(BaseModel):
    verdict: str
    reason: str = Field(default=None)


class HallucinationMetric(BaseMetric):
    def __init__(
        self,
        threshold: float = 0.5,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        run_async: bool = False,
        strict_mode: bool = False,
    ):
        self.threshold = 0 if strict_mode else threshold
        if isinstance(model, DeepEvalBaseLLM):
            self.model = model
        else:
            self.model = GPTModel(model=model)
        self.evaluation_model = self.model.get_model_name()
        self.include_reason = include_reason
        self.run_async = run_async
        self.strict_mode = strict_mode

    def measure(self, test_case: LLMTestCase):
        if (
            test_case.input is None
            or test_case.actual_output is None
            or test_case.context is None
        ):
            raise ValueError("Input, actual output, or context cannot be None")
        with metrics_progress_context(
            self.__name__, self.evaluation_model, self.strict_mode
        ):
            if self.run_async:
                loop = get_or_create_event_loop()
                self.verdicts: List[HallucinationVerdict] = (
                    loop.run_until_complete(
                        self._a_generate_verdicts(
                            test_case.actual_output, test_case.context
                        )
                    )
                )
                self.score = self._generate_score()
                self.reason = loop.run_until_complete(self._a_generate_reason())

            else:
                self.verdicts: List[HallucinationVerdict] = (
                    self._generate_verdicts(
                        test_case.actual_output, test_case.context
                    )
                )
                self.score = self._generate_score()
                self.reason = self._generate_reason()

            self.success = self.score <= self.threshold
            capture_metric_type(self.__name__)
            return self.score

    async def _a_generate_reason(self):
        if self.include_reason is False:
            return None

        factual_alignments = []
        contradictions = []
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "no":
                factual_alignments.append(verdict.reason)
            else:
                contradictions.append(verdict.reason)

        prompt: dict = HallucinationTemplate.generate_reason(
            factual_alignments=factual_alignments,
            contradictions=contradictions,
            score=format(self.score, ".2f"),
        )

        if self.run_async:
            res = await self.model.a_generate(prompt)
        else:
            res = self.model.generate(prompt)

        return res

    def _generate_reason(self):
        loop = get_or_create_event_loop()
        return loop.run_until_complete(self._a_generate_reason())

    def _generate_score(self) -> float:
        number_of_verdicts = len(self.verdicts)
        if number_of_verdicts == 0:
            return 0

        hallucination_count = 0

        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "no":
                hallucination_count += 1

        score = hallucination_count / number_of_verdicts

        return 1 if self.strict_mode and score > self.threshold else score

    async def _a_generate_verdicts(
        self, actual_output: str, contexts: List[str]
    ) -> List[HallucinationVerdict]:
        verdicts: List[HallucinationVerdict] = []

        if self.run_async:
            tasks = [
                self._a_generate_verdict(actual_output, context)
                for context in contexts
            ]
            results = await asyncio.gather(*tasks)
            verdicts.extend(results)
        else:
            prompt = HallucinationTemplate.generate_verdicts(
                actual_output=actual_output, contexts=contexts
            )
            res = self.model.generate(prompt)
            data = trimAndLoadJson(res)
            verdicts = [
                HallucinationVerdict(**item) for item in data["verdicts"]
            ]

        return verdicts

    def _generate_verdicts(
        self, actual_output: str, contexts: List[str]
    ) -> List[HallucinationVerdict]:
        loop = get_or_create_event_loop()
        return loop.run_until_complete(
            self._a_generate_verdicts(actual_output, contexts)
        )

    async def _a_generate_verdict(
        self, actual_output: str, context: str
    ) -> HallucinationVerdict:
        print("generating verdict")
        #######################################
        ### Generate verdicts for [context] ###
        #######################################
        prompt = HallucinationTemplate.generate_verdicts(
            actual_output=actual_output, contexts=[context]
        )

        if self.run_async:
            res = await self.model.a_generate(prompt)
        else:
            res = self.model.generate(prompt)

        data = trimAndLoadJson(res)
        # verdicts length will always be 1
        final_verdicts = [
            HallucinationVerdict(**item) for item in data["verdicts"]
        ]

        return final_verdicts[0]

    def _generate_verdict(
        self, actual_output: str, context: str
    ) -> HallucinationVerdict:
        print("generating verdict")
        loop = get_or_create_event_loop()
        return loop.run_until_complete(
            self._a_generate_verdict(actual_output, context)
        )

    def is_successful(self) -> bool:
        self.success = self.score <= self.threshold
        return self.success

    @property
    def __name__(self):
        return "Hallucination"
