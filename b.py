from typing import List
from deepeval.metrics.base_metric import BaseMetric
from deepeval.metrics.utils import check_llm_test_case_params
from deepeval.test_case import LLMTestCase, ConversationalTestCase
from deepeval import confident_evaluate, evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    BiasMetric,
    FaithfulnessMetric,
    ConversationCompletenessMetric,
)
from deepeval.test_case.llm_test_case import LLMTestCaseParams

test_case = ConversationalTestCase(
    turns=[
        LLMTestCase(
            input="Message input", actual_output="Message actual output"
        )
    ]
)
test_case2 = ConversationalTestCase(
    turns=[
        LLMTestCase(
            input="Message input", actual_output="Message actual output"
        )
    ]
)

required_params: List[LLMTestCaseParams] = [
    LLMTestCaseParams.INPUT,
    LLMTestCaseParams.ACTUAL_OUTPUT,
    LLMTestCaseParams.RETRIEVAL_CONTEXT,
]


class FakeMetric(BaseMetric):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase, _show_indicator: bool):
        check_llm_test_case_params(test_case, required_params, self)
        self.score = 1
        self.success = self.score >= self.threshold
        self.reason = "This metric looking good!"
        return self.score

    async def a_measure(self, test_case: LLMTestCase, _show_indicator: bool):
        check_llm_test_case_params(test_case, required_params, self)
        self.score = 1
        self.success = self.score >= self.threshold
        self.reason = "This metric looking good!"
        return self.score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Fake"


evaluate(
    test_cases=[
        LLMTestCase(
            input="Message input", actual_output="Message actual output"
        ),
        LLMTestCase(
            input="Message input 2",
            actual_output="Message actual output 2",
            retrieval_context=[""],
        ),
    ],
    metrics=[FakeMetric(), FaithfulnessMetric()],
    skip_on_missing_params=True,
    ignore_errors=True,
)

# confident_evaluate(experiment_name="Convo", test_cases=[test_case])


# evaluate(
#     test_cases=[
#         LLMTestCase(
#             input="Message input", actual_output="Message actual output"
#         )
#     ],
#     metrics=[
#         AnswerRelevancyMetric(),
#         BiasMetric(),
#         FaithfulnessMetric(),
#         ConversationCompletenessMetric(),
#     ],
#     run_async=True,
#     ignore_errors=True,
# )
