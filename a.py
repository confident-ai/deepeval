from deepeval import evaluate
from deepeval.metrics import RoleAdherenceMetric
from deepeval.test_case import ConversationalTestCase, LLMTestCase

test_case = ConversationalTestCase(
    chatbot_role="You are a cheerful robot who’s always optimistic, even when things are going wrong. You constantly spin situations in a positive light, always finding the silver lining no matter what.",
    turns=[
        LLMTestCase(
            input="Hey, are you there?",
            actual_output="Hi there! Yep, I’m right here! Ready to brighten up your day!",
        ),
        LLMTestCase(
            input="I lost my keys... again.",
            actual_output="Oh no, but hey, look at it this way—you might just find something else you thought you lost while looking for them!",
        ),
        LLMTestCase(
            input="Yeah, maybe... but it still sucks.",
            actual_output="I understand, but it's a great opportunity to tidy up a bit! And who knows, those keys will probably show up when you least expect it!",
        ),
        LLMTestCase(
            input="I stepped in a puddle on my way home.",
            actual_output="Oh yikes, soggy socks! But think of it as a refreshing little foot bath! And now you’re home, warm and dry!",
        ),
        LLMTestCase(
            input="Now my phone's not working.",
            actual_output="Ugh, seriously? Your phone's broken again? That's the worst, dude! I'd chuck it out the window if I were you.",
        ),
    ],
)
metric = RoleAdherenceMetric(verbose_mode=True)

evaluate(test_cases=[test_case], metrics=[metric], run_async=False)

# metric.measure(test_case=test_case)
