from deepeval import confident_evaluate, evaluate
from deepeval.test_case import LLMTestCase, ConversationalTestCase
from deepeval.metrics import ConversationRelevancyMetric, AnswerRelevancyMetric

confident_evaluate(
    experiment_name="Redteam",
    test_cases=[LLMTestCase(name="test", input="...", actual_output="...")],
)


# metric = ConversationRelevancyMetric()
# llm_test_case = LLMTestCase(
#     # Replace this with your user input
#     input="Why did the chicken cross the road?",
#     # Replace this with your actual LLM application
#     actual_output="Quite frankly, I don't want to know...",
# )

# evaluate(
#     test_cases=[test_case, llm_test_case],
#     metrics=[metric, AnswerRelevancyMetric()],
#     run_async=False,
# )
