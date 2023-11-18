"""Asserting conceptual similarity
"""
from typing import Optional

from deepeval.singleton import Singleton
from deepeval.test_case import LLMTestCase
from deepeval.utils import cosine_similarity
from deepeval.progress_context import progress_context
from deepeval.metrics.base_metric import BaseMetric


class ConceptualSimilarityMetric(BaseMetric, metaclass=Singleton):
    """basic implementation of ConceptualSimilarityMetric"""

    def __init__(
        self,
        model_name: Optional[str] = "sentence-transformers/all-mpnet-base-v2",
        minimum_score: float = 0.7,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        with progress_context(
            "Downloading Conceptual Similarity model (may take up to 2 minutes if running for the first time)..."
        ):
            self.model = SentenceTransformer(self.model_name).eval()
        self.minimum_score = minimum_score

    def _vectorize(self, text_a: str, text_b: str):
        vectors = self.model.encode([text_a, text_b])
        return vectors

    def measure(self, test_case: LLMTestCase) -> float:
        if test_case.actual_output is None or test_case.expected_output is None:
            raise ValueError("Output or expected_output cannot be None")

        vectors = self._vectorize(
            test_case.actual_output, test_case.expected_output
        )
        self.score = cosine_similarity(vectors[0], vectors[1])
        self.success = self.score >= self.minimum_score
        return float(self.score)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return "Conceptual Similarity"
