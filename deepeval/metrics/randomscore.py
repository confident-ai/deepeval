import random
from .metric import Metric
from ..test_case import LLMTestCase
from ..singleton import Singleton


class RandomMetric(Metric, metaclass=Singleton):
    def __init__(self, minimum_score: float = 0.3):
        self.minimum_score = minimum_score

    def measure(self, test_case: LLMTestCase):
        self.score = random.random()
        self.success = self.score >= self.minimum_score
        return self.score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Random"
