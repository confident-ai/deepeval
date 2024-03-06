from typing import Optional, Union, List
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel, Field

from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric
from deepeval.utils import trimAndLoadJson
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
        multithreading: bool = True,
        strict_mode: bool = False,
    ):
        self.threshold = 1 if strict_mode else threshold
        if isinstance(model, DeepEvalBaseLLM):
            self.model = model
        else:
            self.model = GPTModel(model=model)
        self.evaluation_model = self.model.get_model_name()
        self.include_reason = include_reason
        self.multithreading = multithreading
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
            self.verdicts: List[HallucinationVerdict] = self._generate_verdicts(
                test_case.actual_output, test_case.context
            )
            hallucination_score = self._generate_score()
            self.reason = self._generate_reason(hallucination_score)
            self.success = hallucination_score <= self.threshold
            self.score = hallucination_score
            capture_metric_type(self.__name__)
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
        number_of_verdicts = len(self.verdicts)
        if number_of_verdicts == 0:
            return 0

        hallucination_count = 0

        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "no":
                hallucination_count += 1

        score = hallucination_count / number_of_verdicts

        return 1 if self.strict_mode and score > self.threshold else score

    def _generate_verdicts(
        self, actual_output: str, contexts: List[str]
    ) -> List[HallucinationVerdict]:
        verdicts: List[HallucinationVerdict] = []

        if self.multithreading:
            lock = Lock()
            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(
                        self._generate_verdict,
                        actual_output,
                        context,
                        verdicts,
                        lock,
                    ): context
                    for context in contexts
                }

                for future in as_completed(futures):
                    future.result()
        else:
            prompt = HallucinationTemplate.generate_verdicts(
                actual_output=actual_output, contexts=contexts
            )
            res = self.model(prompt)
            data = trimAndLoadJson(res)
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
        data = trimAndLoadJson(res)

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
