from deepeval.singleton import Singleton
from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric
import numpy as np


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class AnswerRelevancyModel(metaclass=Singleton):
    def __init__(self):
        from sentence_transformers import SentenceTransformer

        # Load the model
        self.model = SentenceTransformer(
            "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
        )

    def encode(self, text):
        return self.model.encode(text)


class CrossEncoderAnswerRelevancyModel(metaclass=Singleton):
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base"):
        from sentence_transformers.cross_encoder import CrossEncoder

        self.model = CrossEncoder(model_name)

    def encode(self, question: str, answer: str):
        scores = self.model.predict([[question, answer]])
        return softmax(scores[0])[2]


class AnswerRelevancyMetric(BaseMetric, metaclass=Singleton):
    def __init__(
        self, minimum_score: float = 0.5, model_type: str = "cross_encoder"
    ):
        self.minimum_score = minimum_score
        if model_type == "cross_encoder":
            self.model = CrossEncoderAnswerRelevancyModel()
        else:
            self.model = AnswerRelevancyModel()

    def __call__(self, test_case: LLMTestCase):
        score = self.measure(test_case.input, test_case.actual_output)
        self.success = score > self.minimum_score
        return score

    def measure(self, test_case: LLMTestCase) -> float:
        from sentence_transformers import util

        if test_case.input is None or test_case.actual_output is None:
            raise ValueError("query and output cannot be None")

        if isinstance(self.model, CrossEncoderAnswerRelevancyModel):
            score = self.model.encode(test_case.input, test_case.actual_output)
        else:
            docs = [test_case.actual_output]
            # Encode query and documents
            query_emb = self.model.encode(test_case.input)
            doc_emb = self.model.encode(docs)
            # Compute dot score between query and all document embeddings
            scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()
            score = scores[0]

        self.success = score > self.minimum_score
        # Log answer relevancy
        self.score = score
        return score

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return "Answer Relevancy"
