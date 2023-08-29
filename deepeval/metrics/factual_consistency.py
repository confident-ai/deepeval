from ..singleton import Singleton
from ..utils import softmax, chunk_text
from .metric import Metric


class FactualConsistencyMetric(Metric, metaclass=Singleton):
    def __init__(
        self,
        minimum_score: float = 0.6,
        model_name: str = "cross-encoder/nli-deberta-v3-large",
    ):
        # We use a smple cross encoder model
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)
        self.minimum_score = minimum_score

    def measure(self, output: str, context: str):
        context_list = chunk_text(context)
        max_score = 0
        for c in context_list:
            scores = self.model.predict([(context, output), (output, context)])
            # https://huggingface.co/cross-encoder/nli-deberta-base
            # label_mapping = ["contradiction", "entailment", "neutral"]
            softmax_scores = softmax(scores)
            score = softmax_scores[0][1]
            if score > max_score:
                max_score = score

            second_score = softmax_scores[1][1]
            if second_score > max_score:
                max_score = second_score

        self.success = max_score > self.minimum_score
        print({"success": self.success, "score": max_score})
        self.log(
            success=self.success,
            score=max_score,
            metric_name="Factual Consistency",
            output=output,
            expected_output=context,
        )
        return max_score

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
