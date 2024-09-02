from deepeval import confident_evaluate, evaluate
from deepeval.test_case import LLMTestCase, ConversationalTestCase, Message
from deepeval.metrics import ConversationRelevancyMetric

# confident_evaluate(
#     experiment_name="Redteam",
#     test_cases=[LLMTestCase(name="test", input="...", actual_output="...")],
# )

metric = ConversationRelevancyMetric()
llm_test_case = LLMTestCase(
    # Replace this with your user input
    input="Why did the chicken cross the road?",
    # Replace this with your actual LLM application
    actual_output="Quite frankly, I don't want to know...",
)

test_case = ConversationalTestCase(
    messages=[Message(llm_test_case=llm_test_case)]
)

evaluate(test_cases=[test_case], metrics=[metric])
