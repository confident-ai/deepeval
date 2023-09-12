from ..singleton import Singleton
from ..test_case import LLMTestCase
from ..utils import softmax
from .metric import Metric
from sentence_transformers import CrossEncoder


class EntailmentScoreMetric(Metric, metaclass=Singleton):
    def __init__(
        self,
        minimum_score: float = 0.6,
        model_name: str = "cross-encoder/nli-deberta-base",
    ):
        # We use a smple cross encoder model

        self.model = CrossEncoder(model_name)
        self.minimum_score = minimum_score

    def measure(self, test_case: LLMTestCase):
        scores = self.model.predict([(test_case.query, test_case.output)])
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
