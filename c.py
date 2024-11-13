from deepeval.metrics import JsonCorrectnessMetric

from deepeval.metrics.faithfulness.schema import FaithfulnessVerdict, Verdicts
from deepeval.test_case import LLMTestCase

metric = JsonCorrectnessMetric(expected_schema=Verdicts, verbose_mode=True)

answer = """{\n"verdicts": [\n{\n"verdict": "yes"\n},\n{\n    "verdict": "no",\n    "reason": "blah blah"\n},'
    '\n{\n    "verdict": "yes",\n    "reason":null \n}\n]\n}"""

test_case = LLMTestCase(input="...", actual_output=answer)

metric.measure(test_case=test_case)