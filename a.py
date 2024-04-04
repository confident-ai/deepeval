from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

test_case = LLMTestCase(
    input="What is the primary difference between a comet and an asteroid?",
    actual_output="ok",
    # retrieval_context=[one, two, three],
)
metric = FaithfulnessMetric()

metric.measure(test_case)
print(metric.score)
print(metric.reason)
print(metric.error)
print(metric.is_successful())
