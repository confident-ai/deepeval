import os
from deepeval.singleton import Singleton
from deepeval.test_case import LLMTestCase
from deepeval.utils import chunk_text, softmax
from deepeval.metrics.metric import Metric
from deepeval.run_test import assert_test
from deepeval.progress_context import progress_context
from sentence_transformers import CrossEncoder


class FactualConsistencyModel(metaclass=Singleton):
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-large"):
        # We use a smple cross encoder model
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
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

        with progress_context(
            "Downloading FactualConsistencyModel (may take up to 2 minutes if running for the first time)..."
        ):
            self.model = FactualConsistencyModel(model_name)
        self.minimum_score = minimum_score

    def measure(self, test_case: LLMTestCase, return_all_scores: bool = False):
        if test_case.output is None or test_case.context is None:
            raise ValueError("Output or context cannot be None")

        context_list = []
        if isinstance(test_case.context, str):
            context_list.extend(chunk_text(test_case.context))
        elif isinstance(test_case.context, list):
            for context in test_case.context:
                context_list.extend(chunk_text(context))
        else:
            raise ValueError("Context must be a string or a list of strings")

        max_context_score = 0
        for c in context_list:
            score = self.model.predict(c, test_case.output)
            if score > max_context_score:
                max_context_score = score

        ground_truth_score = 0
        if test_case.expected_output and test_case.expected_output != "-":
            ground_truth_score = self.model.predict(
                test_case.expected_output, test_case.output
            )

        self.success = (
            max_context_score > self.minimum_score
            or ground_truth_score > self.minimum_score
        )
        self.context_score = max_context_score
        self.ground_truth_score = ground_truth_score
        if return_all_scores:
            return {
                "context_score": self.context_score,
                "ground_truth_score": self.ground_truth_score,
                "max_score": max(max_context_score, ground_truth_score),
            }

        return max(max_context_score, ground_truth_score)

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
