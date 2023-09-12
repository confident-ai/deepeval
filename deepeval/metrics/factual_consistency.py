from ..singleton import Singleton
from ..test_case import LLMTestCase
from ..utils import chunk_text, softmax
from .metric import Metric
from ..run_test import assert_test

from sentence_transformers import CrossEncoder


class FactualConsistencyModel(metaclass=Singleton):
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-large"):
        # We use a smple cross encoder model

        self.model = CrossEncoder(model_name)

    def predict(self, text_a: str, text_b: str):
        scores = self.model.predict([(text_a, text_b), (text_b, text_a)])
        # https://huggingface.co/cross-encoder/nli-deberta-base
        # label_mapping = ["contradiction", "entailment", "neutral"]
        softmax_scores = softmax(scores)
        score = softmax_scores[0][1]
        second_score = softmax_scores[1][1]
        return max(score, second_score)


class FactualConsistencyMetric(Metric, metaclass=Singleton):
    def __init__(
        self,
        minimum_score: float = 0.6,
        model_name: str = "cross-encoder/nli-deberta-v3-large",
    ):
        # For Crossencoder model, move to singleton to avoid re-instantiating
        self.model = FactualConsistencyModel(model_name)
        self.minimum_score = minimum_score

    def measure(self, test_case: LLMTestCase):
        if test_case.output is None or test_case.context is None:
            raise ValueError("Output or context cannot be None")

        context_list = chunk_text(test_case.context)
        max_score = 0
        for c in context_list:
            score = self.model.predict(c, test_case.output)
            if score > max_score:
                max_score = score

        self.success = max_score > self.minimum_score
        print({"success": self.success, "score": max_score})
        return max_score

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return "Factual Consistency"


def assert_factual_consistency(
    output: str, context: str, minimum_score: float = 0.3
):
    """Assert that the output is factually consistent with the context."""

    metric = FactualConsistencyMetric(minimum_score=minimum_score)
    test_case = LLMTestCase(output=output, context=context)
    assert_test(test_case, [metric])
