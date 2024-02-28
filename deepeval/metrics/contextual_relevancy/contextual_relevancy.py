from typing import Optional, List, Union
from pydantic import BaseModel, Field
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

from deepeval.utils import trimAndLoadJson
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
    sentence: str = Field(default=None)


class ContextualRelevancyMetric(BaseMetric):
    def __init__(
        self,
        threshold: float = 0.5,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        multithreading: bool = True,
    ):
        self.threshold = threshold
        if isinstance(model, DeepEvalBaseLLM):
            self.model = model
        else:
            self.model = GPTModel(model=model)
        self.evaluation_model = self.model.get_model_name()
        self.include_reason = include_reason
        self.multithreading = multithreading

    def measure(self, test_case: LLMTestCase) -> float:
        if (
            test_case.input is None
            or test_case.actual_output is None
            or test_case.retrieval_context is None
        ):
            raise ValueError(
                "Input, actual output, or retrieval context cannot be None"
            )
        with metrics_progress_context(self.__name__, self.evaluation_model):
            self.verdicts_list: List[List[ContextualRelevancyVerdict]] = (
                self._generate_verdicts_list(
                    test_case.input, test_case.retrieval_context
                )
            )
            contextual_recall_score = self._generate_score()

            self.reason = self._generate_reason(
                test_case.input, contextual_recall_score
            )

            self.success = contextual_recall_score >= self.threshold
            self.score = contextual_recall_score
            capture_metric_type(self.__name__)
            return self.score

    def _generate_reason(self, input: str, score: float):
        if self.include_reason is False:
            return None

        irrelevant_sentences = []
        for index, verdicts in enumerate(self.verdicts_list):
            for verdict in verdicts:
                if verdict.verdict.strip().lower() == "no":
                    data = {"Node": index + 1, "Sentence": verdict.sentence}
                    irrelevant_sentences.append(data)

        prompt: dict = ContextualRelevancyTemplate.generate_reason(
            input=input,
            irrelevant_sentences=irrelevant_sentences,
            score=format(score, ".2f"),
        )

        res = self.model(prompt)
        return res

    def _generate_score(self):
        irrelevant_sentences = 0
        total_sentence_count = 0
        for verdicts in self.verdicts_list:
            for verdict in verdicts:
                total_sentence_count += 1
                if verdict.verdict.lower() == "no":
                    irrelevant_sentences += 1

        if total_sentence_count == 0:
            return 0

        return (
            total_sentence_count - irrelevant_sentences
        ) / total_sentence_count

    def _generate_verdicts(
        self,
        text: str,
        context: str,
        verdicts_list: List[List[ContextualRelevancyVerdict]],
        lock: Lock,
    ):
        prompt = ContextualRelevancyTemplate.generate_verdicts(
            text=text, context=context
        )

        res = self.model(prompt)
        data = trimAndLoadJson(res)
        verdicts = [
            ContextualRelevancyVerdict(**item) for item in data["verdicts"]
        ]

        with lock:
            verdicts_list.append(verdicts)

    def _generate_verdicts_list(
        self, text: str, retrieval_context: List[str]
    ) -> List[List[ContextualRelevancyVerdict]]:
        verdicts_list: List[List[ContextualRelevancyVerdict]] = []

        if self.multithreading:
            lock = Lock()

            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(
                        self._generate_verdicts,
                        text,
                        context,
                        verdicts_list,
                        lock,
                    ): context
                    for context in retrieval_context
                }

                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as exc:
                        raise exc
        else:
            for context in retrieval_context:
                prompt = ContextualRelevancyTemplate.generate_verdicts(
                    text=text, context=context
                )

                res = self.model(prompt)
                data = trimAndLoadJson(res)
                verdicts = [
                    ContextualRelevancyVerdict(**item)
                    for item in data["verdicts"]
                ]
                verdicts_list.append(verdicts)

        return verdicts_list

    def is_successful(self) -> bool:
        self.success = self.score >= self.threshold
        return self.success

    @property
    def __name__(self):
        return "Contextual Relevancy"
