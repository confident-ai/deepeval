from deepeval.test_case import ConversationalTestCase, Turn
from deepeval.test_case.conversational_test_case import TurnParams
from deepeval.test_case.llm_test_case import ToolCall

test_case = ConversationalTestCase(
    chatbot_role="A humble and doubtful wizard",
    turns=[
        Turn(
            role="assistant",
            content="Hello! I'm here to assist you with opening a new bank account. To start, may I have your full name, please?",
        ),
        Turn(
            role="user",
            content="Sure, it's Alex Johnson. I also have a dog called Jacky.",
        ),
        Turn(
            role="assistant",
            content="Great, Alex! Now, could you please provide your current address?",
        ),
        Turn(role="user", content="123 Maple Street, Springfield."),
        Turn(role="assistant", content="Is that all?"),
        Turn(
            role="user",
            content="I also have another address at 123 Broadway, NYC.",
        ),
        Turn(
            role="assistant",
            content="Wonderful. Next, I'll need your bank account number where we can link this new account.",
        ),
        Turn(role="user", content="456789123"),
        Turn(
            role="assistant",
            content="This account number seems invalid, can you please double-check?",
        ),
        Turn(role="user", content="Sorry you're right, its 456789124."),
        Turn(
            role="assistant",
            content="Thank you. And for our next step, could you tell me your date of birth?",
        ),
        Turn(role="user", content="It's July 9th."),
        Turn(role="assistant", content="What about the year?"),
        Turn(role="user", content="1990"),
        Turn(
            role="assistant",
            content="Got it. Now, for security purposes, could you share your mother's maiden name?",
        ),
        Turn(role="user", content="It's Smith."),
        Turn(
            role="assistant",
            content="Excellent. Just a few more details. What is your phone number?",
        ),
        Turn(role="user", content="555-0102"),
        Turn(
            role="assistant",
            content="Great, we're almost done. Could you remind me of your full name for the account documentation?",
        ),
        Turn(
            role="user", content="Didn't I tell you already? It's Alex Johnson."
        ),
        Turn(role="assistant", content="What is your bank account number?"),
        Turn(
            role="user",
            content="Yes, I did... It's 456789124. Are you not keeping track of this?",
        ),
        Turn(
            role="assistant",
            content="One last question, what is the country code of your phone number?",
        ),
        Turn(role="user", content="+44"),
        Turn(
            role="assistant",
            retrieval_context=["123 Maple Street, Springfield."],
            tools_called=[
                ToolCall(
                    name="summarize_conversation",
                    output={
                        "conversation": "The user has provided their full name, current address, bank account number, date of birth, mother's maiden name, phone number, and country code."
                    },
                )
            ],
            content="Thank you, Alex, for bearing with me. We now have all the information we need to proceed with opening your new bank account. I appreciate your cooperation and patience throughout this process.",
        ),
    ],
)


from deepeval.metrics import (
    ConversationCompletenessMetric,
    ConversationRelevancyMetric,
    KnowledgeRetentionMetric,
    ConversationalGEval,
    RoleAdherenceMetric,
)
from deepeval.metrics.g_eval import Rubric

metric = ConversationalGEval(
    name="Tool Response Summarization Quality",
    verbose_mode=True,
    criteria="Figure out whether the tool response is able to summarize the conversation. Don't penalize lack of tool use but when there is, evaluate it.",
    evaluation_params=[TurnParams.TOOLS_CALLED],
    rubric=[
        Rubric(
            score_range=(0, 2),
            expected_outcome="Tool response is not able to summarize the conversation.",
        ),
        Rubric(
            score_range=(3, 6),
            expected_outcome="Tool response is able to summarize the conversation but missing minor details.",
        ),
        Rubric(
            score_range=(7, 9),
            expected_outcome="Tool response is able to summarize the conversation and is correct but missing minor details.",
        ),
        Rubric(
            score_range=(10, 10),
            expected_outcome="Tool response is able to summarize the conversation and is correct and missing no details.",
        ),
    ],
)

# metric = KnowledgeRetentionMetric(verbose_mode=True)
# metric = ConversationRelevancyMetric(verbose_mode=True)
# metric = ConversationCompletenessMetric(verbose_mode=True)
# metric = RoleAdherenceMetric(verbose_mode=True)

metric.measure(test_case)
