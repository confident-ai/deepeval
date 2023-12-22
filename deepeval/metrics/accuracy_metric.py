from deepeval.singleton import Singleton
from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric
from deepeval.scorer import Scorer

class ExactMatchAccuracyMetric(BaseMetric, metaclass=Singleton):
    def __init__(self, minimum_score: float = 0.5) -> None:
        self.minimum_score = minimum_score
    
    def __call__(self, test_case: LLMTestCase) -> float:
        score = Scorer.exact_match_score(target=test_case.expected_output, prediction=test_case.actual_output)
        self.success = score > self.minimum_score
        return self.score 
    
    def is_successful(self) -> bool:
        return self.success
    
    @property
    def __name__(self):
        return "ExactMatchAccuracyMetric"