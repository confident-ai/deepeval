from typing import Optional

from ..test_case import LLMTestCase
from .metric import Metric


class CohereRerankerMetric(Metric):
    def __init__(self, api_key: str, minimum_score: float = 0.5):
        try:
            import cohere

            self.cohere_client = cohere.Client(api_key)
        except Exception as e:
            print(e)
            print("Run `pip install cohere`.")
        self.minimum_score = minimum_score

    def __call__(self, output, expected_output, query: Optional[str] = "-"):
        score = self.measure(output, expected_output)
        success = False
        if score > self.minimum_score:
            success = True
        return score

    def measure(self, test_case: LLMTestCase):
        reranked_results = self.cohere_client.rerank(
            query=test_case.query,
            documents=[test_case.output],
            top_n=1,
            model="rerank-english-v2.0",
        )
        score = reranked_results[0].relevance_score
        self.success = score > self.minimum_score
        return score

    def is_successful(self) -> bool:
        return self.success
