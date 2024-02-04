from deepeval.singleton import Singleton
from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric
from deepeval.scorer import Scorer


class NERMetric(BaseMetric):
    def __init__(
        self,
        minimum_score: float = 0.5,
    ):
        self.minimum_score = minimum_score

    def measure(self, test_case: LLMTestCase):
        # if test_case.actual_output is None:
        #     raise ValueError("Output cannot be None")
        score = Scorer.ner_score(test_case.input,test_case.actual_output)

        self.success = score >= self.minimum_score
        self.score = score
        return score

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return "NER"
