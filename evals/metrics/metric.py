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
