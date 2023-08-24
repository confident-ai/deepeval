from typing import Optional
from .metric import Metric
from ..singleton import Singleton
from ..utils import cosine_similarity


class BertScoreMetric(Metric, metaclass=Singleton):
    """basic implementation of BertScoreMetric"""

    def __init__(
        self,
        model_name: Optional[str] = "sentence-transformers/all-mpnet-base-v2",
        minimum_score: float = 0.7,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name).eval()
        self.minimum_score = minimum_score

    def _vectorize(self, text_a: str, text_b: str):
        vectors = self.model.encode([text_a, text_b])
        return vectors

    def measure(self, a: str, b: str):
        vectors = self._vectorize(a, b)
        self.score = cosine_similarity(vectors[0], vectors[1])
        return self.score

    def is_successful(self) -> bool:
        return self.score >= self.minimum_score

    @property
    def __name__(self):
        return "BertScore"
