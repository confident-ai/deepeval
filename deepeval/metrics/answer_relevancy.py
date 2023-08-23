import asyncio
import numpy as np
from .metric import Metric


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class AnswerRelevancy(Metric):
    def __init__(self, minimum_score: bool = 0.5):
        self.minimum_score = minimum_score
        from sentence_transformers import CrossEncoder

        self.encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2")

    def __call__(self, query: str, answer: str):
        score = self.measure(query, answer)
        if self._is_send_okay():
            asyncio.create_task(
                self._send_to_server(
                    metric_score=score,
                    query=query,
                    output=answer,
                    metric_name=self.__name__,
                    implementation_id="",
                )
            )
        return score

    def measure(self, query, answer: str) -> float:
        score = self.encoder.predict([query, answer])
        score = sigmoid(score)
        self.success = score > self.minimum_score
        return score

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return "Answer Relevancy"
