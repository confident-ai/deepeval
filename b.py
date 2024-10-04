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


from deepeval.metrics import GEval

correctness_metric = GEval(
    name="Correctness",
    criteria="Determine whether the actual output is factually correct based on the expected output.",
    # NOTE: you can only provide either criteria or evaluation_steps, and not both
    evaluation_steps=[
        "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
        "You should also heavily penalize omission of detail",
        "Vague language, or contradicting OPINIONS, are OK",
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
)

evaluate(
    test_cases=[
        LLMTestCase(
            input="Message input number 1!",
            actual_output="Message actual output number 1...",
        ),
        LLMTestCase(
            input="Message input 2, this is just a test",
            actual_output="Message actual output 2, this is just a test",
        ),
    ],
    metrics=[
        correctness_metric,
        AnswerRelevancyMetric(),
        BiasMetric(),
        ConversationCompletenessMetric(),
    ],
    # throttle_value=10,
    max_concurrent=1,
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
