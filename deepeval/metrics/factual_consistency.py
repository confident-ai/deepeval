from sentence_transformers import CrossEncoder
from ..singleton import Singleton
from ..utils import softmax
from .metric import Metric


class FactualConsistencyMetric(Metric, metaclass=Singleton):
    def __init__(
        self,
        minimum_score: float = 0.6,
        model_name: str = "cross-encoder/nli-deberta-base",
    ):
        # We use a smple cross encoder model
        self.model = CrossEncoder(model_name)
        self.minimum_score = minimum_score

    def measure(self, output: str, context: str):
        scores = self.model.predict([(output, context)])
        # https://huggingface.co/cross-encoder/nli-deberta-base
        # label_mapping = ["contradiction", "entailment", "neutral"]
        score = softmax(scores)[0][1]
        self.success = score > self.minimum_score
        self.log(
            success=self.success,
            score=score,
            metric_name=self.__name__,
            output=output,
            expected_output=context,
        )
        return score

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return "Factual Consistency"


def assert_factual_consistency(output: str, context: str, minimum_score: float = 0.3):
    """Assert that the output is factually consistent with the context."""

    metric = FactualConsistencyMetric(minimum_score=minimum_score)
    score = metric(context, output)
    assert metric.is_successful(), metric.__class__.__name__ + " was unsuccessful."
