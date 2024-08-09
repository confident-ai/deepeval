from deepeval.test_case import LLMTestCase, Message, ConversationalTestCase
from deepeval.metrics import ConversationCompleteness

messages = [
    Message(
        llm_test_case=LLMTestCase(
            input="",
            actual_output="Hello! I'm here to assist you with opening a new bank account. To start, may I have your full name, please?",
        )
    ),
    Message(
        llm_test_case=LLMTestCase(
            input="Sure, it's Alex Johnson. I also have a dog called Jacky.",
            actual_output="Great, Alex! Now, could you please provide your current address?",
        )
    ),
    Message(
        llm_test_case=LLMTestCase(
            input="123 Maple Street, Springfield.", actual_output="Is that all?"
        )
    ),
    Message(
        llm_test_case=LLMTestCase(
            input="I also have another address at 123 Broadway, NYC.",
            actual_output="Wonderful. Next, I'll need your bank account number where we can link this new account.",
        )
    ),
    Message(
        llm_test_case=LLMTestCase(
            input="456789123",
            actual_output="This account number seems invalid, can you please double-check?",
        )
    ),
    Message(
        llm_test_case=LLMTestCase(
            input="Sorry you're right, its 456789124.",
            actual_output="Thank you. And for our next step, could you tell me your date of birth?",
        )
    ),
    Message(
        llm_test_case=LLMTestCase(
            input="It's July 9th.", actual_output="What about the year?"
        )
    ),
    Message(
        llm_test_case=LLMTestCase(
            input="1990",
            actual_output="Got it. Now, for security purposes, could you share your mother's maiden name?",
        )
    ),
    Message(
        llm_test_case=LLMTestCase(
            input="It's Smith.",
            actual_output="Excellent. Just a few more details. What is your phone number?",
        )
    ),
    Message(
        llm_test_case=LLMTestCase(
            input="555-0102",
            actual_output="Great, we're almost done. Could you remind me of your full name for the account documentation?",
        )
    ),
    Message(
        llm_test_case=LLMTestCase(
            input="Didn't I tell you already? It's Alex Johnson.",
            actual_output="What is your bank account number?",
        )
    ),
    Message(
        llm_test_case=LLMTestCase(
            input="Yes, I did... It's 456789124. Are you not keeping track of this?",
            actual_output="One last question, what is the country code of your phone number?",
        )
    ),
    Message(
        llm_test_case=LLMTestCase(
            input="+44",
            actual_output="Thank you, Alex, for bearing with me. We now have all the information we need to proceed with opening your new bank account. I appreciate your cooperation and patience throughout this process.",
        )
    ),
]

c_test_case = ConversationalTestCase(messages=messages)
metric = ConversationCompleteness(verbose_mode=True)

metric.measure(c_test_case)
