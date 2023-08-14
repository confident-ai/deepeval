import random
from .metric import Metric


class RandomMetric(Metric):
    def __init__(self, minimum_score: float = 0.5):
        self.minimum_score = minimum_score

    def measure(self, a, b):
        self.success = random.random() >= self.minimum_score
        return self.success

    def is_sucessful(self):
        return self.success
