"""Available metrics. The best metric that
you want is Cohere's reranker metric.
"""
import random

from abc import abstractmethod
from ..utils import softmax


class Metric:
    def __call__(self, *args, **kwargs):
        return self.measure(*args, **kwargs)

    @abstractmethod
    def measure(self, a, b):
        pass

    @abstractmethod
    def is_successful(self) -> bool:
        return False


class ConstantMetric(Metric):
    def measure(self, a, b):
        return 0.5

    def is_successful(self):
        return True


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


class EntailmentScore(Metric):
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-base"):
        # We use a smple cross encoder model
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def measure(self, a: str, b: str):
        scores = self.model.predict([(a, b)])
        # https://huggingface.co/cross-encoder/nli-deberta-base
        # label_mapping = ["contradiction", "entailment", "neutral"]
        return softmax(scores)[0][1]
