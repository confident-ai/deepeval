import asyncio
from .metric import Metric
from ..utils import softmax
from typing import Optional


class EntailmentScoreMetric(Metric):
    def __init__(
        self,
        minimum_score: float = 0.6,
        model_name: str = "cross-encoder/nli-deberta-base",
    ):
        # We use a smple cross encoder model
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)
        self.minimum_score = minimum_score

    def measure(self, a: str, b: str):
        scores = self.model.predict([(a, b)])
        # https://huggingface.co/cross-encoder/nli-deberta-base
        # label_mapping = ["contradiction", "entailment", "neutral"]
        score = softmax(scores)[0][1]
        self.success = score > self.minimum_score
        return score

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return "Entailment"

    def __call__(self, output, expected_output, query: Optional[str] = "-"):
        score = self.measure(output, expected_output)
        asyncio.create_task(
            self._send_to_server(
                metric_score=score,
                metric_name=self.__name__,
                query=query,
                output=output,
                expected_output=expected_output,
            )
        )
        return score
