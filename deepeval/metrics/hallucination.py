import json
from typing import Optional, Union, List
from threading import Thread, Lock
from pydantic import BaseModel, Field

from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric
from deepeval.utils import trimToJson
from deepeval.metrics.templates import HallucinationTemplate
from deepeval.models import GPTModel, DeepEvalBaseModel
from deepeval.progress_context import metrics_progress_context


class HallucinationVerdict(BaseModel):
    verdict: str
    reason: str = Field(default=None)


class HallucinationMetric(BaseMetric):
    def __init__(
        self,
        threshold: float = 0.5,
        model: Optional[Union[str, DeepEvalBaseModel]] = None,
        include_reason: bool = True,
        multithreading: bool = True,
    ):
        self.threshold = threshold
        if isinstance(model, DeepEvalBaseModel):
            self.model = model
        else:
            self.model = GPTModel(model=model)
        self.evaluation_model = self.model.get_model_name()
        self.include_reason = include_reason
        self.multithreading = multithreading

    def measure(self, test_case: LLMTestCase):
        if (
            test_case.input is None
            or test_case.actual_output is None
            or test_case.context is None
        ):
            raise ValueError("Input, actual output, or context cannot be None")
        with metrics_progress_context(self.__name__, self.evaluation_model):
            self.verdicts: List[HallucinationVerdict] = self._generate_verdicts(
                test_case.actual_output, test_case.context
            )
            hallucination_score = self._generate_score()
            self.reason = self._generate_reason(hallucination_score)
            self.success = hallucination_score <= self.threshold
            self.score = hallucination_score
            return self.score

    def _generate_reason(self, score):
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
            score=format(score, ".2f"),
        )

        res = self.model(prompt)
        return res

    def _generate_score(self) -> float:
        total = len(self.verdicts)
        hallucination_count = 0
        if total == 0:
            return 0

        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "no":
                hallucination_count += 1

        return hallucination_count / total

    def _generate_verdicts(
        self, actual_output: str, contexts: str
    ) -> List[HallucinationVerdict]:
        verdicts: List[HallucinationVerdict] = []
        threads = []
        lock = Lock()

        if self.multithreading:
            for context in contexts:
                thread = Thread(
                    target=self._generate_verdict,
                    args=(actual_output, context, verdicts, lock),
                )
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()
        else:
            prompt = HallucinationTemplate.generate_verdicts(
                actual_output=actual_output, contexts=contexts
            )
            res = self.model(prompt)
            json_output = trimToJson(res)
            data = json.loads(json_output)
            verdicts = [
                HallucinationVerdict(**item) for item in data["verdicts"]
            ]

        return verdicts

    def _generate_verdict(
        self,
        actual_output: str,
        context: str,
        verdicts: List[HallucinationVerdict],
        lock: Lock,
    ) -> HallucinationVerdict:
        #######################################
        ### Generate verdicts for [context] ###
        #######################################
        prompt = HallucinationTemplate.generate_verdicts(
            actual_output=actual_output, contexts=[context]
        )
        res = self.model(prompt)
        json_output = trimToJson(res)
        data = json.loads(json_output)

        # verdicts length will always be 1
        final_verdicts = [
            HallucinationVerdict(**item) for item in data["verdicts"]
        ]

        with lock:
            for final_verdict in final_verdicts:
                verdicts.append(final_verdict)

    def is_successful(self) -> bool:
        self.success = self.score <= self.threshold
        return self.success

    @property
    def __name__(self):
        return "Hallucination"
