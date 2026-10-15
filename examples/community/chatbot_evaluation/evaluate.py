"""
Simple RAG evaluation example using DeepEval.
Runs Answer Relevancy metric on sample outputs.
"""

import json
import os
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "outputs.json")

with open(DATA_PATH) as f:
    data = json.load(f)

metric = AnswerRelevancyMetric(threshold=0.7)

print("\n--- DeepEval Results ---\n")

for item in data:

    test_case = LLMTestCase(
        input=item["input"], actual_output=item["actual_output"]
    )

    score = metric.measure(test_case)

    print(f"Input: {item['input']}")
    print(f"Score: {score}")
    print("-" * 40)
