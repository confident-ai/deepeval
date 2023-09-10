import numpy as np
from ..test_case import TestCase
from .metric import Metric


class AnswerRelevancyMetric(Metric):
    def __init__(self, minimum_score: bool = 0.5):
        from sentence_transformers import SentenceTransformer, util

        self.minimum_score = minimum_score

        # Load the model
        self.model = SentenceTransformer(
            "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
        )

    def __call__(self, test_case: TestCase):
        score = self.measure(test_case.query, test_case.output)
        success = score > self.minimum_score
        if self._is_send_okay():
            self._send_to_server(
                metric_score=float(score),
                query=test_case.query,
                output=test_case.output,
                metric_name=self.__name__,
                success=bool(success),
            )
        return score

    def measure(self, test_case: TestCase) -> float:
        from sentence_transformers import util

        docs = [test_case.output]

        # Encode query and documents
        query_emb = self.model.encode(test_case.query)
        doc_emb = self.model.encode(docs)

        # Compute dot score between query and all document embeddings
        scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()
        score = scores[0]
        self.success = score > self.minimum_score
        # Log answer relevancy
        self.log(
            success=self.success,
            score=score,
            metric_name=self.__name__,
            query=test_case.query,
            output=test_case.output,
        )
        return score

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return "Answer Relevancy"


def assert_answer_relevancy(query: str, output: str, minimum_score: float = 0.5):
    metric = AnswerRelevancyMetric(minimum_score=minimum_score)
    score = metric(query=query, output=output)
    assert metric.is_successful(), (
        metric.__class__.__name__ + " was unsuccessful - " + str(score)
    )
