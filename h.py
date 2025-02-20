from deepeval.metrics import FaithfulnessMetric
from deepeval import confident_evaluate
from deepeval.test_case import LLMTestCase

test_case = LLMTestCase(input="...", actual_output="...")

confident_evaluate(
    metric_collection="Default Collection", test_cases=[test_case]
)
