import time

from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval.evaluate import evaluate

test_cases = [
    LLMTestCase(
        input=f"What is {i} times 2?",
        actual_output=f"{i} times 2 is {i * 2}.",
        retrieval_context=[f"Multiplying {i} by 2 gives {i * 2}."],
    )
    for i in range(50)
]

start = time.perf_counter()
evaluate(
    test_cases=test_cases,
    metrics=[
        AnswerRelevancyMetric(eval_mode="system_one"),
        FaithfulnessMetric(eval_mode="system_one"),
        ContextualRelevancyMetric(eval_mode="system_one"),
    ],
)
print(f"Took {time.perf_counter() - start:.2f}s")


test_cases = [
    LLMTestCase(
        input=f"What is {i} times 2?",
        actual_output=f"{i} times 2 is {i * 2}.",
        retrieval_context=[f"Multiplying {i} by 2 gives {i * 2}."],
    )
    for i in range(50)
]

start = time.perf_counter()
evaluate(
    test_cases=test_cases,
    metrics=[
        AnswerRelevancyMetric(eval_mode="llm"),
        FaithfulnessMetric(eval_mode="llm"),
        ContextualRelevancyMetric(eval_mode="llm"),
    ],
)
print(f"Took {time.perf_counter() - start:.2f}s")
