import asyncio
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

# Define the test cases
test_case1 = LLMTestCase(
    input="The chicken crossed the road, why?",
    actual_output="Because he felt like it",
)
test_case2 = LLMTestCase(
    input="Knock knock, who's there?", actual_output="The chicken"
)

# Define the metric
metric1 = AnswerRelevancyMetric(verbose_mode=False)

# # Asynchronous function to measure relevancy concurrently
# async def measure_relevancy():
#     await metric1.a_measure(test_case1, _show_indicator=False)
#     print(metric1.statements)
#     print("All measurements are done.")

# # Run the asynchronous function and print after completion
# asyncio.run(measure_relevancy())
# print("This is printed after all asynchronous operations are complete.")


evaluate([test_case1], [metric1])
