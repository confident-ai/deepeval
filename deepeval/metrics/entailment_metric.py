import asyncio
from sentence_transformers import CrossEncoder
from typing import Optional
from ..utils import softmax
from ..singleton import Singleton
from .metric import Metric


class EntailmentScoreMetric(Metric, metaclass=Singleton):
    def __init__(
        self,
        minimum_score: float = 0.6,
        model_name: str = "cross-encoder/nli-deberta-base",
    ):
        # We use a smple cross encoder model
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
        success = False
        if score > self.minimum_score:
            success = True
        asyncio.create_task(
            self._send_to_server(
                metric_score=score,
                metric_name=self.__name__,
                query=query,
                output=output,
                expected_output=expected_output,
                success=success,
            )
        )
        return score
