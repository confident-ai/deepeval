from deepeval.singleton import Singleton
from deepeval.test_case import LLMTestCase
from deepeval.metrics.base_metric import BaseMetric
from deepeval.utils import chunk_text
from deepeval.scorer import Scorer


class FactualConsistencyMetric(BaseMetric, metaclass=Singleton):
    def __init__(
        self,
        minimum_score: float = 0.6,
        model_name: str = "cross-encoder/nli-deberta-v3-large",
    ):
        self.model_name = model_name
        self.minimum_score = minimum_score

    def measure(self, test_case: LLMTestCase):
        if test_case.actual_output is None or test_case.context is None:
            raise ValueError("Output or context cannot be None")

        context_list = []
        if isinstance(test_case.context, str):
            context_list.extend(chunk_text(test_case.context))
        elif isinstance(test_case.context, list):
            for context in test_case.context:
                context_list.extend(chunk_text(context))
        else:
            raise ValueError("Context must be a string or a list of strings")

        score = Scorer.factual_consistency_score(
            contexts=context_list,
            prediction=test_case.actual_output,
            model=self.model_name,
        )
        self.score = score
        self.success = score > self.minimum_score
        return score

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return "Factual Consistency"
