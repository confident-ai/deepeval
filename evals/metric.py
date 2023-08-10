"""Available metrics. The best metric that
you want is Cohere's reranker metric.
"""
import random
import os
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


class CohereRerankerMetric(Metric):
    def __init__(self, api_key: str):
        try:
            import cohere

            self.cohere_client = cohere.Client(api_key)
        except Exception as e:
            print(e)
            print("Make sure to install Cohere.")

    def measure(self, a, b):
        reranked_results = self.cohere_client.rerank(
            query=a,
            documents=[b],
            top_n=1,
            model="rerank-english-v2.0",
        )
        score = reranked_results[0].relevance_score
        return score
