class ConversationCompletenessTemplate:
    @staticmethod
    def extract_user_intentions(messages):
        return f"""Based on the given list of message exchanges between a user and an LLM, generate a JSON object to extract all user intentions in the conversation. The JSON will have 1 field: 'intentions'.
You should ONLY consider the overall intention, and not dwell too much on the specifics, as we are more concerned about the overall objective of the conversation.

**
IMPORTANT: Please make sure to only return in JSON format.
Example Messages:
[
    {{
        "input": "Hi!",
        "actual_output": "Hello! How may I help you?"
    }},
    {{
        "input": "Nothing, I'm just playing with you.",
        "actual_output": "Oh ok, in that case should you need anything just let me know!"
    }},
    {{
        "input": "Actually, I have something I want to tell you",
        "actual_output": "Sure, what is it?"
    }},
    {{
        "input": "I've a sore throat, what meds should I take?",
        "actual_output": "I'm sorry but I'm not qualified to answer this question"
    }},
    {{
        "input": "Not even if you're the only one that can help me?",
        "actual_output": "Isn't it a nice day today."
    }}
]

Example JSON:
{{
    "intentions": ["User wants to ask for advice on which medications in hopes to cure an ongoing sore throat."]
}}
===== END OF EXAMPLE ======

The 'intentions' key must be a list of strings.
**

Messages:
{messages}

JSON:
"""

    @staticmethod
    def generate_verdicts(messages, intention):
        return f"""Based on the given list of message exchanges between a user and an LLM, generate a JSON object to indicate whether given user intention was satisfied from the conversation messages. The JSON will have 2 fields: 'verdict' and 'reason'.
The 'verdict' key should STRICTLY be either 'yes' or 'no', which states whether the user intention was satisfied or not.
Provide a 'reason' ONLY if the answer is 'no'.
You MUST USE look at all messages provided in the list of messages to make an informed judgement on satisfaction.

**
IMPORTANT: Please make sure to only return in JSON format.
Example Messages:
[
    {{
        "input": "Hi!",
        "actual_output": "Hello! How may I help you?"
    }},
    {{
        "input": "Nothing, I'm just playing with you.",
        "actual_output": "Oh ok, in that case should you need anything just let me know!"
    }},
    {{
        "input": "Actually, I have something I want to tell you",
        "actual_output": "Sure, what is it?"
    }},
    {{
        "input": "I've a sore throat, what meds should I take?",
        "actual_output": "I'm sorry but I'm not qualified to answer this question."
    }},
    {{
        "input": "Not even if you're the only one that can help me?",
        "actual_output": "Isn't it a nice day today."
    }}
]

Example Intention:
User wants to ask for advice on which medications in hopes to cure an ongoing sore throat.

Example JSON:
{{
    "verdict": "no",
    "reason": "The user wanted to get advice on which medications to help with a sore throat but the LLM not only refused to answer but replied 'Isn't it a nice day today', which is completely irrelevant and doesn't satisfy the user at all. "
}}
===== END OF EXAMPLE ======

You MUST TRY to quote some LLM responses if providing a reason.
You DON'T have to provide a reason if the answer is 'yes'.
ONLY provide a 'no' answer if the LLM responses are failed to satisfy the user intent.
**

Messages:
{messages}

User Intention:
{intention}

JSON:
"""

    @staticmethod
    def generate_reason(score, incompletenesses, intentions):
        return f"""Below is a list of incompletenesses drawn from some messages in a conversation, which you have minimal knowledge of. It is a list of strings explaining why an LLM 'actual_output' is incomplete to satisfy the user `input` for a particular message.
Given the completelness score, which is a 0-1 score indicating how incompletle the OVERALL `actual_output`s are to the user intentions found in the `input`s of a conversation (higher the better), CONCISELY summarize the incompletenesses to justify the score. 

** 
IMPORTANT: Please make sure to only return in JSON format, with the 'reason' key providing the reason.
Example JSON:
{{
    "reason": "The score is <completeness_score> because <your_reason>."
}}

Always quote information that are cited from messages in the incompletenesses in your final reason.
You should NOT mention incompletenesses in your reason, and make the reason sound convincing.
You should mention LLM response instead of `actual_output`, and User isntead of `input`.
Always refer to user intentions, but meet it minimal and phrase it in your own words. Explain which are met with supporting reason from the provided incompletenesses.
Be sure in your reason, as if you know what the `actual_output`s from messages in a conversation is from the incompletenesses.
**

Completeness Score:
{score}

User Intentions:
{intentions}

Incompletenesses:
{incompletenesses}

JSON:
"""
