import copy
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval import evaluate

metric = AnswerRelevancyMetric(threshold=0.9)
test_case = LLMTestCase(input="ok", actual_output="ok")


evaluate([test_case], [metric])
