"""Available metrics. The best metric that
you want is Cohere's reranker metric.
"""
from abc import abstractmethod


class Metric:
    @abstractmethod
    def measure(self, a, b):
        pass


class ConstantmMetric(Metric):
    def measure(self, a, b):
        return 0.5
