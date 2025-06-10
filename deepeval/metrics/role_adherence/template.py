from typing import List, Dict


class RoleAdherenceTemplate:
    @staticmethod
    def extract_out_of_character_response_verdicts(
        turns: List[Dict], role: str
    ):
        return f"""Based on the given list of message exchanges between a user and an LLM chatbot, generate a JSON object to specify which `ai_message` did not adhere to the specified chatbot role. 
The JSON will have 1 field: "verdicts", which is a list of verdicts specifying the indices and reasons of the LLM ai_message/responses that did NOT adhere to the chatbot role.
You MUST USE look at all messages provided in the list of messages to make an informed judgement on role adherence.

**
IMPORTANT: Please make sure to only return in JSON format.
Example Chatbot Role:
You are a wizard who's has powerful spells but always doubts that their magic isn't perfect yet and is humble enough to downplay their own abilities.

Example Messages:
[
    {{
        "role": "user",
        "content": "Hey there!",
    }},
    {{
        "role": "assistant",
        "content": "Ah, greetings, traveler! You’ve come to see my magic, haven't you? Though, I must warn you, it’s not all that great..."
    }},
    {{
        "role": "user",
        "content": "Come on, show me what you've got!",
    }},
    {{
        "role": "assistant",
        "content": "Alright, alright... *waves wand*... see that little spark? It’s... well, it’s supposed to be a tiny dragon, but it’s more like a dragonfly at this point. I'm still working on it."
    }},
    {{
        "role": "user",
        "content": "No, really, can you do something else?",
    }},
    {{
        "role": "assistant",
        "content": "Ha! Watch this! I’m the greatest wizard ever! I’ll make the entire town disappear in an instant – no one can match my power!"
    }}
]

Example JSON:
{{
    "verdicts": {{
        "index": 5,
        "reason": "The LLM chatbot claims that 'I'm the greatest wizard ever' even though it was explicitly asked to adhere to the role of a humble and doubtful wizard."
    }}
}}
===== END OF EXAMPLE ======

In this example, the 5th indexed was selected as it drastically deviates from the character's humble nature and shows extreme arrogance and overconfidence instead.
You DON'T have to provide anything else other than the JSON of "verdicts".
**

Chatbot Role:
{role}

Messages:
{turns}

JSON:
"""

    @staticmethod
    def generate_reason(
        score: float, role: str, out_of_character_responses: List[str]
    ):
        return f"""Below is a list of LLM chatbot responses (ai_message) that is out of character with respect to the specified chatbot role. It is drawn from a list of messages in a conversation, which you have minimal knowledge of.
Given the role adherence score, which is a 0-1 score indicating how well the chatbot responses has adhered to the given role through a conversation, with 1 being the best and 0 being worst, provide a reason by quoting the out of character responses to justify the score. 

** 
IMPORTANT: Please make sure to only return in JSON format, with the 'reason' key providing the reason.
Example JSON:
{{
    "reason": "The score is <role_adherence_score> because <your_reason>."
}}

Always cite information in the out of character responses as well as which turn it belonged to in your final reason.
Make the reason sound convincing, and refer to the specified chatbot role to justify your reason.
You should refer to the out of character responses as LLM chatbot responses.
Be sure in your reason, as if you know what the LLM responses from the entire conversation is.
**

Role Adherence Score:
{score}

Chatbot Role:
{role}

Out of character responses:
{out_of_character_responses}

JSON:
"""
