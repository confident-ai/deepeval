from deepeval.test_case import LLMTestCase, Message, ConversationalTestCase
from deepeval import confident_evaluate, evaluate
from deepeval.metrics import AnswerRelevancyMetric, BiasMetric

test_case = ConversationalTestCase(
    messages=[
        Message(
            llm_test_case=LLMTestCase(
                input="Message input", actual_output="Message actual output"
            )
        )
    ]
)
test_case2 = ConversationalTestCase(
    messages=[
        Message(
            llm_test_case=LLMTestCase(
                input="Message input", actual_output="Message actual output"
            )
        )
    ]
)

# confident_evaluate(experiment_name="Redteam", test_cases=[test_case, LLMTestCase(input="ok", actual_output="what?")])

evaluate(
    test_cases=[test_case, test_case2],
    metrics=[AnswerRelevancyMetric(), BiasMetric()],
    run_async=True,
)
