import asyncio
import numpy as np
from .metric import Metric
from sentence_transformers import SentenceTransformer, util


class AnswerRelevancy(Metric):
    def __init__(self, minimum_score: bool = 0.5):
        self.minimum_score = minimum_score

        # Load the model
        self.model = SentenceTransformer(
            "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
        )

    def __call__(self, query: str, answer: str):
        score = self.measure(query, answer)
        success = score > self.minimum_score
        if self._is_send_okay():
            asyncio.create_task(
                self._send_to_server(
                    metric_score=score,
                    query=query,
                    output=answer,
                    metric_name=self.__name__,
                    success=success,
                )
            )
        return score

    def measure(self, query, answer: str) -> float:
        docs = [answer]

        # Encode query and documents
        query_emb = self.model.encode(query)
        doc_emb = self.model.encode(docs)

        # Compute dot score between query and all document embeddings
        scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()
        score = scores[0]
        return score

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return "Answer Relevancy"
