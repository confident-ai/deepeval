from deepeval.test_case import LLMTestCase, ConversationalTestCase
from deepeval import confident_evaluate, evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    BiasMetric,
    FaithfulnessMetric,
    ConversationCompletenessMetric,
)

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

# confident_evaluate(experiment_name="Redteam", test_cases=[test_case, LLMTestCase(input="ok", actual_output="what?")])

print(test_case.turns)

[print(turn) for index, turn in enumerate(test_case.turns)]

evaluate(
    test_cases=[test_case],
    metrics=[
        AnswerRelevancyMetric(),
        BiasMetric(),
        FaithfulnessMetric(),
        ConversationCompletenessMetric(),
    ],
    run_async=True,
    ignore_errors=True,
)
