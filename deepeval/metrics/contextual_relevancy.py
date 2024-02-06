from typing import Optional, List, Union
from pydantic import BaseModel, Field
from threading import Thread, Lock
import json
from langchain_core.language_models import BaseChatModel

from deepeval.utils import trimToJson
from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric
from deepeval.models import GPTModel, DeepEvalBaseModel
from deepeval.metrics.templates import ContextualRelevancyTemplate
from deepeval.progress_context import metrics_progress_context


class ContextualRelevancyVerdict(BaseModel):
    verdict: str
    sentence: str = Field(default=None)


class ContextualRelevancyMetric(BaseMetric):
    def __init__(
        self,
        threshold: float = 0.5,
        model: Optional[Union[str, DeepEvalBaseModel, BaseChatModel]] = None,
        include_reason: bool = True,
    ):
        self.threshold = threshold
        if isinstance(model, DeepEvalBaseModel):
            self.model = model
        else:
            self.model = GPTModel(model=model)
        self.evaluation_model = self.model.get_model_name()
        self.include_reason = include_reason

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
            self.verdicts_list: List[
                List[ContextualRelevancyVerdict]
            ] = self._generate_verdicts_list(
                test_case.input, test_case.retrieval_context
            )
            contextual_recall_score = self._generate_score()

            self.reason = self._generate_reason(
                test_case.input, contextual_recall_score
            )

            self.success = contextual_recall_score >= self.threshold
            self.score = contextual_recall_score

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
        json_output = trimToJson(res)
        data = json.loads(json_output)
        verdicts = [
            ContextualRelevancyVerdict(**item) for item in data["verdicts"]
        ]

        with lock:
            verdicts_list.append(verdicts)

    def _generate_verdicts_list(
        self, text: str, retrieval_context: List[str]
    ) -> List[List[ContextualRelevancyVerdict]]:
        verdicts_list: List[List[ContextualRelevancyVerdict]] = []
        threads = []
        lock = Lock()
        for context in retrieval_context:
            thread = Thread(
                target=self._generate_verdicts,
                args=(text, context, verdicts_list, lock),
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        return verdicts_list

    def is_successful(self) -> bool:
        self.success = self.score >= self.threshold
        return self.success

    @property
    def __name__(self):
        return "Contextual Relevancy"
