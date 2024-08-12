from deepeval.test_case import LLMTestCase, Message, ConversationalTestCase
from deepeval.metrics import ConversationCompleteness, AnswerRelevancyMetric
from deepeval import evaluate

messages = [
    Message(
        llm_test_case=LLMTestCase(
            input="+44",
            actual_output="Thank you, Alex, for bearing with me. We now have all the information we need to proceed with opening your new bank account. I appreciate your cooperation and patience throughout this process.",
        ),
        should_evaluate=False,
    ),
]

c_test_case = ConversationalTestCase(messages=messages)
metric = ConversationCompleteness(verbose_mode=True)


evaluate(
    test_cases=[
        LLMTestCase(
            input="+44",
            actual_output="Thank you, Alex, for bearing with me. We now have all the information we need to proceed with opening your new bank account. I appreciate your cooperation and patience throughout this process.",
        ),
        c_test_case,
        LLMTestCase(
            input="+55",
            actual_output="Thank you, Alex, for bearing with me. We now have all the information we need to proceed with opening your new bank account. I appreciate your cooperation and patience throughout this process.",
        ),
    ],
    metrics=[metric, AnswerRelevancyMetric()],
)
