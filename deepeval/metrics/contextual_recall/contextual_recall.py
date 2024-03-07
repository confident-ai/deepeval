from typing import Optional, List, Union
from pydantic import BaseModel, Field

from deepeval.utils import trimAndLoadJson, get_or_create_event_loop
from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric
from deepeval.models import GPTModel, DeepEvalBaseLLM
from deepeval.metrics.contextual_recall.template import ContextualRecallTemplate
from deepeval.progress_context import metrics_progress_context
from deepeval.telemetry import capture_metric_type


class ContextualRecallVerdict(BaseModel):
    verdict: str
    reason: str = Field(default=None)


class ContextualRecallMetric(BaseMetric):
    def __init__(
        self,
        threshold: float = 0.5,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        run_async: bool = True,
        strict_mode: bool = False,
    ):
        self.threshold = 1 if strict_mode else threshold
        if isinstance(model, DeepEvalBaseLLM):
            self.model = model
        else:
            self.model = GPTModel(model=model)
        self.evaluation_model = self.model.get_model_name()
        self.include_reason = include_reason
        self.run_async = run_async
        self.strict_mode = strict_mode

    def measure(self, test_case: LLMTestCase) -> float:
        if (
            test_case.input is None
            or test_case.actual_output is None
            or test_case.retrieval_context is None
            or test_case.expected_output is None
        ):
            raise ValueError(
                "Input, actual output, expected output, or retrieval context cannot be None"
            )
        with metrics_progress_context(
            self.__name__, self.evaluation_model, self.strict_mode
        ):
            if self.run_async:
                loop = get_or_create_event_loop()
                self.verdicts: List[ContextualRecallVerdict] = (
                    loop.run_until_complete(
                        self._a_generate_verdicts(
                            test_case.expected_output,
                            test_case.retrieval_context,
                        )
                    )
                )
                self.score = self._generate_score()
                self.reason = loop.run_until_complete(
                    self._a_generate_reason(test_case.expected_output)
                )
            else:
                self.verdicts: List[ContextualRecallVerdict] = (
                    self._generate_verdicts(
                        test_case.expected_output, test_case.retrieval_context
                    )
                )
                self.score = self._generate_score()
                self.reason = self._generate_reason(test_case.expected_output)

            self.success = self.score >= self.threshold
            capture_metric_type(self.__name__)
            return self.score

    async def _a_generate_reason(self, expected_output: str):
        if self.include_reason is False:
            return None

        supportive_reasons = []
        unsupportive_reasons = []
        for verdict in self.verdicts:
            if verdict.verdict.lower() == "yes":
                supportive_reasons.append(verdict.reason)
            else:
                unsupportive_reasons.append(verdict.reason)

        prompt = ContextualRecallTemplate.generate_reason(
            expected_output=expected_output,
            supportive_reasons=supportive_reasons,
            unsupportive_reasons=unsupportive_reasons,
            score=format(self.score, ".2f"),
        )

        if self.run_async:
            res = await self.model.a_generate(prompt)
        else:
            res = self.model.generate(prompt)

        return res

    def _generate_reason(self, expected_output: str):
        loop = get_or_create_event_loop()
        return loop.run_until_complete(self._a_generate_reason(expected_output))

    def _generate_score(self):
        number_of_verdicts = len(self.verdicts)
        if number_of_verdicts == 0:
            return 0

        justified_sentences = 0
        for verdict in self.verdicts:
            if verdict.verdict.lower() == "yes":
                justified_sentences += 1

        score = justified_sentences / number_of_verdicts

        return 0 if self.strict_mode and score < self.threshold else score

    async def _a_generate_verdicts(
        self, expected_output: str, retrieval_context: List[str]
    ) -> List[ContextualRecallVerdict]:
        prompt = ContextualRecallTemplate.generate_verdicts(
            expected_output=expected_output, retrieval_context=retrieval_context
        )
        if self.run_async:
            res = await self.model.a_generate(prompt)
        else:
            res = self.model.generate(prompt)

        data = trimAndLoadJson(res)
        verdicts = [
            ContextualRecallVerdict(**item) for item in data["verdicts"]
        ]
        return verdicts

    def _generate_verdicts(
        self, expected_output: str, retrieval_context: List[str]
    ) -> List[ContextualRecallVerdict]:
        loop = get_or_create_event_loop()
        return loop.run_until_complete(
            self._a_generate_verdicts(expected_output, retrieval_context)
        )

    def is_successful(self) -> bool:
        self.success = self.score >= self.threshold
        return self.success

    @property
    def __name__(self):
        return "Contextual Recall"
