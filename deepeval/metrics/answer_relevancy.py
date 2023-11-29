from deepeval.singleton import Singleton
from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric
from deepeval.scorer import Scorer


class AnswerRelevancyMetric(BaseMetric, metaclass=Singleton):
    def __init__(
        self, minimum_score: float = 0.5, model_type: str = "cross_encoder"
    ):
        self.minimum_score, self.model_type = minimum_score, model_type

    def __call__(self, test_case: LLMTestCase):
        score = self.measure(test_case.input, test_case.actual_output)
        self.success = score > self.minimum_score
        return score

    def measure(self, test_case: LLMTestCase) -> float:
        answer_relevancy_score = Scorer.answer_relevancy_score(
            predictions=test_case.input,
            target=test_case.actual_output,
            model_type=self.model_type,
        )

        self.success = answer_relevancy_score > self.minimum_score
        self.score = answer_relevancy_score
        return answer_relevancy_score

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return "Answer Relevancy"
