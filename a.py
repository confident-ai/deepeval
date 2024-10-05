from deepeval.test_case import LLMTestCase, ConversationalTestCase
from deepeval.metrics import KnowledgeRetentionMetric

turns = [
    LLMTestCase(
        input="",
        actual_output="Hello! I'm here to assist you with opening a new bank account. To start, may I have your full name, please?",
    ),
    LLMTestCase(
        input="Sure, it's Alex Johnson. I also have a dog called Jacky.",
        actual_output="Great, Alex! Now, could you please provide your current address?",
    ),
    LLMTestCase(
        input="123 Maple Street, Springfield.", actual_output="Is that all?"
    ),
    LLMTestCase(
        input="I also have another address at 123 Broadway, NYC.",
        actual_output="Wonderful. Next, I'll need your bank account number where we can link this new account.",
    ),
    LLMTestCase(
        input="456789123",
        actual_output="This account number seems invalid, can you please double-check?",
    ),
    LLMTestCase(
        input="Sorry you're right, its 456789124.",
        actual_output="Thank you. And for our next step, could you tell me your date of birth?",
    ),
    LLMTestCase(input="It's July 9th.", actual_output="What about the year?"),
    LLMTestCase(
        input="1990",
        actual_output="Got it. Now, for security purposes, could you share your mother's maiden name?",
    ),
    LLMTestCase(
        input="It's Smith.",
        actual_output="Excellent. Just a few more details. What is your phone number?",
    ),
    LLMTestCase(
        input="555-0102",
        actual_output="Great, we're almost done. Could you remind me of your full name for the account documentation?",
    ),
    LLMTestCase(
        input="Didn't I tell you already? It's Alex Johnson.",
        actual_output="What is your bank account number?",
    ),
    LLMTestCase(
        input="Yes, I did... It's 456789124. Are you not keeping track of this?",
        actual_output="One last question, what is the country code of your phone number?",
    ),
    LLMTestCase(
        input="+44",
        actual_output="Thank you, Alex, for bearing with me. We now have all the information we need to proceed with opening your new bank account. I appreciate your cooperation and patience throughout this process.",
    ),
]

test_case = ConversationalTestCase(turns=turns)
metric = KnowledgeRetentionMetric(threshold=0.5, verbose_mode=True)

metric.measure(test_case)
