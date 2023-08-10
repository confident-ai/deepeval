"""Available metrics. The best metric that
you want is Cohere's reranker metric.
"""
import random
from abc import abstractmethod


class Metric:
    @abstractmethod
    def measure(self, a, b):
        pass


class ConstantMetric(Metric):
    def measure(self, a, b):
        return 0.5


class RandomMetric(Metric):
    def measure(self, a, b):
        return random.random()
