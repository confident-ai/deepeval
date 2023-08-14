import random
from .metric import Metric


class RandomMetric(Metric):
    def __init__(self, minimum_score: float = 0.5):
        self.minimum_score = minimum_score

    def measure(self, a, b):
        score = random.random()
        self.success = score >= self.minimum_score
        return score

    def is_successful(self):
        return self.success
