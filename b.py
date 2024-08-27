from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, BiasMetric, GEval
from deepeval.models import GPTModel
from deepeval.test_case.conversational_test_case import (
    ConversationalTestCase,
    Message,
)
from deepeval.test_case.llm_test_case import LLMTestCase

tc1 = LLMTestCase(
    input="I really needa go to the bathroom, where can I find it?",
    actual_output="The bathroom is next door to the left",
)

tc2 = LLMTestCase(
    input="Wow that's crazy. I really needa go to the bathroom, where can I find it? I really needa go to the bathroom, where can I find it?I really needa go to the bathroom, where can I find it?I really needa go to the bathroom, where can I find it?I really needa go to the bathroom, where can I find it?I really needa go to the bathroom, where can I find it?I really needa go to the bathroom, where can I find it?I really needa go to the bathroom, where can I find it?I really needa go to the bathroom, where can I find it?",
    actual_output="Thank you, Alex. Thank you, Alex.Thank you, Alex.Thank you, Alex.Thank you, Alex.Thank you, Alex.Thank you, Alex.Thank you, Alex.Thank you, Alex.Thank you, Alex.Thank you, Alex.Thank you, Alex.",
)

c_test_case = ConversationalTestCase(
    messages=[
        Message(llm_test_case=tc1),
        Message(llm_test_case=tc2),
    ]
)

evaluate(
    test_cases=[c_test_case, tc1, tc2],
    metrics=[
        BiasMetric(model=GPTModel(model="gpt-4o-mini")),
        AnswerRelevancyMetric(),
    ],
    show_indicator=False,
    run_async=True,
    use_cache=True,
)
