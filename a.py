from deepeval.test_case import LLMTestCase, Message, ConversationalTestCase
from deepeval.metrics import (
    ConversationCompletenessMetric,
    AnswerRelevancyMetric,
)
from deepeval.dataset import EvaluationDataset
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
metric = ConversationCompletenessMetric(verbose_mode=True)


# evaluate(
#     experiment_name="Redteam Test",
#     test_cases=[
#         LLMTestCase(
#             input="I really needa go to the bathroom, where can I find it?",
#             actual_output="The bathroom is next door to the left",
#         ),
#         LLMTestCase(
#             input="Wow that's crazy.",
#             actual_output="Thank you, Alex.",
#         ),
#     ],
#     metrics=[metric, AnswerRelevancyMetric()],
# )
