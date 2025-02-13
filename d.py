from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase


test_case = LLMTestCase(
    input="",
    actual_output="The Capital of France is London",
    retrieval_context=["Paris is the Capital of France."],
)

metric = FaithfulnessMetric(model="gpt-4o-mini", verbose_mode=True)
metric.measure(test_case)
