from typing import List, Optional, Union
import json
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor

from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric
from deepeval.utils import trimToJson
from deepeval.models import GPTModel, DeepEvalBaseModel
from deepeval.metrics.templates import FaithfulnessTemplate
from deepeval.progress_context import metrics_progress_context
from deepeval.telemetry import capture_metric_type


class FaithfulnessVerdict(BaseModel):
    verdict: str
    reason: str = Field(default=None)


class FaithfulnessMetric(BaseMetric):
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
            or test_case.retrieval_context is None
        ):
            raise ValueError(
                "Input, actual output, and retrieval context cannot be None"
            )
        with metrics_progress_context(self.__name__, self.evaluation_model):
            if self.multithreading:
                # Use multithreading to generate truths and claims in parallel
                with ThreadPoolExecutor() as executor:
                    future_truths = executor.submit(
                        self._generate_truths, test_case.retrieval_context
                    )
                    future_claims = executor.submit(
                        self._generate_claims, test_case.actual_output
                    )
                    self.truths: List[str] = future_truths.result()
                    self.claims: List[str] = future_claims.result()
            else:
                # Sequential execution
                self.truths: List[str] = self._generate_truths(
                    test_case.retrieval_context
                )
                self.claims: List[str] = self._generate_claims(
                    test_case.actual_output
                )

            self.verdicts: List[FaithfulnessVerdict] = self._generate_verdicts()
            faithfulness_score = self._generate_score()
            self.reason = self._generate_reason(faithfulness_score)
            self.success = faithfulness_score >= self.threshold
            self.score = faithfulness_score
            capture_metric_type(self.__name__)
            return self.score

    def _generate_reason(self, score) -> str:
        if self.include_reason is False:
            return None

        contradictions = []
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "no":
                contradictions.append(verdict.reason)

        prompt: dict = FaithfulnessTemplate.generate_reason(
            contradictions=contradictions,
            score=format(score, ".2f"),
        )

        res = self.model(prompt)
        return res

    def _generate_score(self) -> float:
        total = len(self.verdicts)
        if total == 0:
            return 0
        faithfulness_count = 0
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() != "no":
                faithfulness_count += 1

        return faithfulness_count / total

    def _generate_verdicts(self) -> List[FaithfulnessVerdict]:
        verdicts: List[FaithfulnessVerdict] = []

        prompt = FaithfulnessTemplate.generate_verdicts(
            claims=self.claims, retrieval_context="\n\n".join(self.truths)
        )
        res = self.model(prompt)
        json_output = trimToJson(res)
        data = json.loads(json_output)
        verdicts = [FaithfulnessVerdict(**item) for item in data["verdicts"]]

        return verdicts

    def _generate_truths(self, retrieval_context: str) -> List[str]:
        prompt = FaithfulnessTemplate.generate_claims(
            text="\n\n".join(retrieval_context)
        )
        res = self.model(prompt)
        json_output = trimToJson(res)
        data = json.loads(json_output)

        return data["claims"]

    def _generate_claims(self, actual_output: str) -> List[str]:
        prompt = FaithfulnessTemplate.generate_claims(text=actual_output)
        res = self.model(prompt)
        json_output = trimToJson(res)
        data = json.loads(json_output)

        return data["claims"]

    def is_successful(self) -> bool:
        self.success = self.score >= self.threshold
        return self.success

    @property
    def __name__(self):
        return "Faithfulness"
