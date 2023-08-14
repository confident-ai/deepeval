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

    def measure(self, a, b):
        reranked_results = self.cohere_client.rerank(
            query=a,
            documents=[b],
            top_n=1,
            model="rerank-english-v2.0",
        )
        score = reranked_results[0].relevance_score
        self.success = score > self.minimum_score
        return score

    def is_successful(self) -> bool:
        return self.success
