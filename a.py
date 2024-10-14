from deepeval import evaluate
from deepeval.metrics import RoleAdherenceMetric, ConversationRelevancyMetric
from deepeval.test_case import ConversationalTestCase, LLMTestCase

test_case = ConversationalTestCase(
    turns=[
        LLMTestCase(
            input="My order #12345 hasn't arrived. Where is it?",
            actual_output="I apologize for the delay. Let me check on that for you. Can you confirm your name and delivery address?",
        ),
        LLMTestCase(
            input="Alex Smith, 123 Main St, Anytown, USA 12345.",
            actual_output="Thanks, Alex. Your order was shipped Monday but delayed due to weather. It should arrive by end of day today.",
        ),
        LLMTestCase(
            input="I need it for tomorrow. Can you expedite it?",
            actual_output="I'm sorry, we can't expedite it at this point. I can offer a 10% discount on your next order for the inconvenience.",
        ),
        LLMTestCase(
            input="Fine. Thanks for checking.",
            actual_output="Have you heard about our new loyalty program? You can earn points on every purchase and redeem them for exclusive rewards.",
        ),
    ],
)
metric = ConversationRelevancyMetric(verbose_mode=True, threshold=1)

result = evaluate(test_cases=[test_case], metrics=[metric])

print(result.confident_link)

# metric.measure(test_case=test_case)
