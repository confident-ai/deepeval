from deepeval.test_case import LLMTestCase, Message, ConversationalTestCase
from deepeval import confident_evaluate, evaluate
from deepeval.metrics import AnswerRelevancyMetric, BiasMetric

test_case = ConversationalTestCase(
    messages=[
        Message(llm_test_case=LLMTestCase(input="ok", actual_output="what?"))
    ]
)

# confident_evaluate(experiment_name="Redteam", test_cases=[test_case])

evaluate(
    test_cases=[test_case], metrics=[AnswerRelevancyMetric(), BiasMetric()]
)
