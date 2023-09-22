from abc import abstractmethod

from ..singleton import Singleton
from ..test_case import LLMTestCase
from ..utils import softmax
from sentence_transformers import CrossEncoder


class Metric(metaclass=Singleton):
    # set an arbitrary minimum score that will get over-ridden later
    minimum_score: float = 0
    score: float = 0

    # Measure function signature is subject to be different - not sure
    # how applicable this is - might need a better abstraction
    @abstractmethod
    def measure(self, test_case: LLMTestCase, *args, **kwargs):
        raise NotImplementedError

    def _get_init_values(self):
        # We use this method for sending useful metadata
        init_values = {
            param: getattr(self, param)
            for param in vars(self)
            if isinstance(getattr(self, param), (str, int, float))
        }
        return init_values

    @abstractmethod
    def is_successful(self) -> bool:
        raise NotImplementedError

    @property
    def __name__(self):
        return "Metric"


class EntailmentScoreMetric(Metric):
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-base"):
        # We use a smple cross encoder model

        self.model = CrossEncoder(model_name)

    def measure(self, a: str, b: str):
        scores = self.model.predict([(a, b)])
        # https://huggingface.co/cross-encoder/nli-deberta-base
        # label_mapping = ["contradiction", "entailment", "neutral"]
        return softmax(scores)[0][1]
