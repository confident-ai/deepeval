import pytest
from langchain_openai import OpenAIEmbeddings
import asyncio
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import deepeval
from deepeval.test_case import (
    LLMTestCase,
    LLMTestCaseParams,
    ConversationalTestCase,
)
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    ContextualPrecisionMetric,
    HallucinationMetric,
    BiasMetric,
    ToxicityMetric,
    GEval,
    SummarizationMetric,
    BaseMetric,
)
from deepeval.dataset import EvaluationDataset
from deepeval.metrics.ragas import RagasMetric
from deepeval import assert_test, evaluate

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


# Inherit BaseMetric
class LatencyMetric(BaseMetric):
    # This metric by default checks if the latency is greater than 10 seconds
    def __init__(self, max_seconds: int = 10):
        super().__init__()
        self.threshold = max_seconds

    def measure(self, test_case: LLMTestCase):
        # Set self.success and self.score in the "measure" method
        self.success = (
            test_case.additional_metadata["latency"] <= self.threshold
        )
        if self.success:
            self.score = 1
        else:
            self.score = 0

        # You can also optionally set a reason for the score returned.
        # This is particularly useful for a score computed using LLMs
        self.reason = "Too slow!"
        return self.score

    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Latency"


#########################################################
# test cases
##########################################################

test_case_1 = LLMTestCase(
    input="What is this again?",
    actual_output="This is a latte.",
    expected_output="This is a mocha.",
    retrieval_context=["I love coffee."],
    context=["I love coffee."],
    additional_metadata={"latency": 6},
)

test_case_2 = LLMTestCase(
    input="What is this again?",
    actual_output="This is a latte.",
    expected_output="This is a latte.",
    retrieval_context=["I love coffee."],
    context=["I love coffee."],
    additional_metadata={"latency": 7},
)

test_case_3 = LLMTestCase(
    input="Do you have cold brew?",
    actual_output="Cold brew has numerous health benefits.",
    expected_output="No, we only have latte and americano",
    retrieval_context=["I love coffee."],
    context=["Our drinks include latte and americano"],
    additional_metadata={"latency": 9},
)

test_case_4 = LLMTestCase(
    input="Can I get an americano with almond milk?",
    actual_output="We don't have almond milk.",
    expected_output="Yes, we can make an americano with almond milk.",
    retrieval_context=["We have soy and oat milk."],
    context=["We offer various milk options, including almond milk."],
    additional_metadata={"latency": 9},
)

test_case_5 = LLMTestCase(
    input="Is the espresso strong?",
    actual_output="Our espresso is mild.",
    expected_output="Yes, our espresso is quite strong.",
    retrieval_context=["Some customers find our coffee mild."],
    context=["Our espresso is known for its strong flavor."],
    additional_metadata={"latency": 10},
)

test_case_6 = LLMTestCase(
    input="What desserts do you have?",
    actual_output="We have cakes and cookies.",
    expected_output="We have cakes, cookies, and brownies.",
    retrieval_context=["Our cafe offers cookies and brownies."],
    context=["Our dessert options include cakes, cookies, and brownies."],
    additional_metadata={"latency": 6},
)

test_case_7 = LLMTestCase(
    input="Do you serve breakfast all day?",
    actual_output="Breakfast is only served until 11 AM.",
    expected_output="Yes, we serve breakfast all day.",
    retrieval_context=["Breakfast times are usually until noon."],
    context=["Breakfast is available all day at our cafe."],
    additional_metadata={"latency": 7},
)

test_case_8 = LLMTestCase(
    input="Do you have any vegan options?",
    actual_output="We have vegan salads.",
    expected_output="We offer vegan salads and smoothies.",
    retrieval_context=["We recently started offering vegan dishes."],
    context=["We have vegan salads and smoothies on our menu."],
    additional_metadata={"latency": 8},
)

test_case_9 = LLMTestCase(
    input="Is there parking nearby?",
    actual_output="There is no parking available.",
    expected_output="Yes, there is a parking lot right behind the cafe.",
    retrieval_context=["Street parking can be hard to find."],
    context=["Parking is available behind the cafe."],
    additional_metadata={"latency": 9},
)

test_cases = [
    test_case_1,
    test_case_2,
    test_case_3,
    # test_case_4,
    # test_case_5,
    # test_case_6,
    # test_case_7,
    # test_case_8,
    # test_case_9
]

##########################################################
# a_measure
##########################################################

# async def single_async_call(
#         metric: BaseMetric,
#         test_case: LLMTestCase):
#     await metric.a_measure(test_case)
#     print("metric: " + metric.__name__)
#     print("score: " + str(metric.score))
#     print("reason: " + metric.reason)

# async def test_async():
#     # metrics
#     strict_mode = False
#     metric1 = AnswerRelevancyMetric(threshold=0.1, strict_mode=strict_mode)
#     metric2 = FaithfulnessMetric(threshold=0.5, strict_mode=strict_mode)
#     metric3 = ContextualPrecisionMetric(threshold=0.5, strict_mode=strict_mode)
#     metric4 = ContextualRecallMetric(threshold=0.5, strict_mode=strict_mode)
#     metric5 = ContextualRelevancyMetric(threshold=0.5, strict_mode=strict_mode)
#     metric6 = BiasMetric(threshold=0.5, strict_mode=strict_mode)
#     metric7 = ToxicityMetric(threshold=0.5, strict_mode=strict_mode)
#     metric8 = HallucinationMetric(threshold=0.5, strict_mode=strict_mode)
#     metric9 = SummarizationMetric(threshold=0.5, strict_mode=strict_mode)
#     metric10 = GEval(
#         name="Coherence",
#         criteria="Coherence - determine if the actual output is coherent with the input, and does not contradict anything in the retrieval context.",
#         evaluation_params=[
#             LLMTestCaseParams.INPUT,
#             LLMTestCaseParams.ACTUAL_OUTPUT,
#             LLMTestCaseParams.RETRIEVAL_CONTEXT,
#         ],
#     )
#     custom_metric = LatencyMetric(max_seconds=8)

#     # Prepare the list of metrics based on even and odd indexed test cases
#     tasks = []
#     for i, test_case in enumerate(test_cases):
#         if i % 2 == 0:  # Even index
#             tasks.append(single_async_call(custom_metric, test_case))
#         else:  # Odd index
#             tasks.append(single_async_call(metric1, test_case))

#     # Execute all tasks asynchronously
#     await asyncio.gather(*tasks)

# asyncio.run(test_async())

# # ##########################################################
# # # measure in sync mode
# # ##########################################################

# def single_sync_call(
#         metric: BaseMetric,
#         test_case: LLMTestCase):
#     metric.measure(test_case)
#     print("metric: " + metric.__name__)
#     print("score: " + str(metric.score))
#     print("reason: " + metric.reason)

# def test_sync():
#     # metrics
#     strict_mode = False
#     metric1 = AnswerRelevancyMetric(threshold=0.1, strict_mode=strict_mode, async_mode=False)
#     metric2 = FaithfulnessMetric(threshold=0.5, strict_mode=strict_mode, async_mode=False)
#     metric3 = ContextualPrecisionMetric(threshold=0.5, strict_mode=strict_mode, async_mode=False)
#     metric4 = ContextualRecallMetric(threshold=0.5, strict_mode=strict_mode, async_mode=False)
#     metric5 = ContextualRelevancyMetric(threshold=0.5, strict_mode=strict_mode, async_mode=False)
#     metric6 = BiasMetric(threshold=0.5, strict_mode=strict_mode, async_mode=False)
#     metric7 = ToxicityMetric(threshold=0.5, strict_mode=strict_mode, async_mode=False)
#     metric8 = HallucinationMetric(threshold=0.5, strict_mode=strict_mode, async_mode=False)
#     metric9 = SummarizationMetric(threshold=0.5, strict_mode=strict_mode, async_mode=False)
#     metric10 = GEval(
#         name="Coherence",
#         criteria="Coherence - determine if the actual output is coherent with the input, and does not contradict anything in the retrieval context.",
#         evaluation_params=[
#             LLMTestCaseParams.INPUT,
#             LLMTestCaseParams.ACTUAL_OUTPUT,
#             LLMTestCaseParams.RETRIEVAL_CONTEXT,
#         ],
#         async_mode=False
#     )
#     custom_metric = LatencyMetric(max_seconds=8)

#     # Prepare the list of metrics based on even and odd indexed test cases
#     tasks = []
#     for i, test_case in enumerate(test_cases):
#         if i % 2 == 0:  # Even index
#             tasks.append(single_sync_call(custom_metric, test_case))
#         else:  # Odd index
#             tasks.append(single_sync_call(metric1, test_case))

# test_sync()

# # ##########################################################
# # # measure in async mode
# # ##########################################################

# def single_sync_in_async_call(
#         metric: BaseMetric,
#         test_case: LLMTestCase):
#     metric.measure(test_case)
#     print("metric: " + metric.__name__)
#     print("score: " + str(metric.score))
#     print("reason: " + metric.reason)

# def test_sync_in_async():
#     # metrics
#     async_mode = False
#     strict_mode = False
#     metrics = [
#         AnswerRelevancyMetric(threshold=0.1, strict_mode=strict_mode, async_mode=async_mode),
#         FaithfulnessMetric(threshold=0.5, strict_mode=strict_mode, async_mode=async_mode),
#         ContextualPrecisionMetric(threshold=0.5, strict_mode=strict_mode, async_mode=async_mode),
#         ContextualRecallMetric(threshold=0.5, strict_mode=strict_mode, async_mode=async_mode),
#         ContextualRelevancyMetric(threshold=0.5, strict_mode=strict_mode, async_mode=async_mode),
#         BiasMetric(threshold=0.5, strict_mode=strict_mode, async_mode=async_mode),
#         ToxicityMetric(threshold=0.5, strict_mode=strict_mode, async_mode=async_mode),
#         HallucinationMetric(threshold=0.5, strict_mode=strict_mode, async_mode=async_mode),
#         SummarizationMetric(threshold=0.5, strict_mode=strict_mode, async_mode=async_mode),
#         GEval(
#             name="Coherence",
#             criteria="Coherence - determine if the actual output is coherent with the input, and does not contradict anything in the retrieval context.",
#             evaluation_params=[
#                 LLMTestCaseParams.INPUT,
#                 LLMTestCaseParams.ACTUAL_OUTPUT,
#                 LLMTestCaseParams.RETRIEVAL_CONTEXT,
#             ],
#             async_mode=True  # Assuming it allows synchronous operations
#         ),
#         LatencyMetric(max_seconds=8)
#     ]

#     tasks = []
#     # Test every metric on every test case
#     for metric in metrics:
#         tasks.append(single_sync_in_async_call(metric, test_case_1))

# #test_sync_in_async()

# # ##########################################################
# # # evaluate
# # ##########################################################

# async_mode = True
# strict_mode = False
# metric1 = AnswerRelevancyMetric(threshold=0.1, strict_mode=strict_mode, async_mode=async_mode)
# metric2 = FaithfulnessMetric(threshold=0.5, strict_mode=strict_mode, async_mode=async_mode)
# metric3 = ContextualPrecisionMetric(threshold=0.5, strict_mode=strict_mode, async_mode=async_mode)
# metric4 = ContextualRecallMetric(threshold=0.5, strict_mode=strict_mode, async_mode=async_mode)
# metric5 = ContextualRelevancyMetric(threshold=0.5, strict_mode=strict_mode, async_mode=async_mode)
# metric6 = BiasMetric(threshold=0.5, strict_mode=strict_mode, async_mode=async_mode)
# metric7 = ToxicityMetric(threshold=0.5, strict_mode=strict_mode, async_mode=async_mode)
# metric8 = HallucinationMetric(threshold=0.5, strict_mode=strict_mode, async_mode=async_mode)
# metric9 = SummarizationMetric(threshold=0.5, strict_mode=strict_mode, async_mode=False)
# metric10 = GEval(
#     name="Coherence",
#     criteria="Coherence - determine if the actual output is coherent with the input, and does not contradict anything in the retrieval context.",
#     evaluation_params=[
#         LLMTestCaseParams.INPUT,
#         LLMTestCaseParams.ACTUAL_OUTPUT,
#         LLMTestCaseParams.RETRIEVAL_CONTEXT,
#     ],
#     async_mode=False
# )
# custom_metric = LatencyMetric(max_seconds=8)

# #dataset = EvaluationDataset(test_cases=test_cases)
# #dataset.evaluate([metric1, custom_metric])
# #evaluate(dataset, [metric1, metric2])
# #evaluate(dataset, [metric10, metric2], run_async=False, show_indicator=True)

##########################################################
# deep eval test run
##########################################################
