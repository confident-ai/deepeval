from deepeval.test_case import ConversationalTestCase, LLMTestCase, Message

from deepeval.metrics import AnswerRelevancyMetric, BiasMetric
from deepeval import evaluate


tc1 = LLMTestCase(input="hi again", actual_output="bye")
tc2 = LLMTestCase(input="hi", actual_output="bye")

c_test_case = ConversationalTestCase(
    messages=[Message(llm_test_case=tc2), Message(llm_test_case=tc1)]
)


evaluate(
    test_cases=[c_test_case, tc1],
    metrics=[AnswerRelevancyMetric(), BiasMetric()],
    run_async=False,
)
