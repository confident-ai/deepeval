from deepeval.metrics import AnswerRelevancyMetric, TurnRelevancyMetric
from deepeval.test_case import LLMTestCase, ConversationalTestCase, Turn

metric_1 = AnswerRelevancyMetric(threshold=0.7)
metric_2 = AnswerRelevancyMetric(threshold=0.7)
metric_3 = AnswerRelevancyMetric(threshold=0.7)
test_case = LLMTestCase(
    input="What is the weather in San Francisco?",
    actual_output="The weather in San Francisco is sunny and 70 degrees.",
    expected_output="The weather in San Francisco is sunny and 70 degrees.",
)


import asyncio


async def main():
    await asyncio.gather(
        metric_1.a_measure(test_case),
        metric_2.a_measure(test_case),
        metric_3.a_measure(test_case),
    )


if __name__ == "__main__":
    asyncio.run(main())
