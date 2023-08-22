"""Metric for bias classifier - using the same min score subtraction methodology as the toxic classifier 
Rationale for bias classifier is described here https://arxiv.org/pdf/2208.05777.pdf
1 - Not Biased
0 - Bias
"""


from .metric import Metric
from Dbias.bias_classification import *


class UnBiasedMetric(Metric):
    def __init__(
        self, model_name: str = "original", minimum_score: float = 0.5
    ):  # see paper for rationale https://arxiv.org/pdf/2208.05777.pdf
        self.model_name = model_name
        self.model = classifier(model_name)
        self.minimum_score = minimum_score

    def measure(self, text: str):
        results = self.model.predict(text)
        # sample output
        # [{'label': 'Biased', 'score': 0.9938021898269653}]
        self.success = True
        v = score_value = results[0]["score"]
        if v > 1 - self.minimum_score:
            self.success = False
        return results

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Biased'"


def assert_unbiased(
    text: str, minimum_score: float = 0.5
):  # see paper for rationale https://arxiv.org/pdf/2208.05777.pdf
    metric = UnBiasedMetric(minimum_score=minimum_score)
    score = metric.measure(text)
    assert metric.is_successful(), f"Text is biased - got {score}"
