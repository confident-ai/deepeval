class RoleAdherenceTemplate:
    @staticmethod
    def extract_out_of_character_response_indicies(turns, role):
        return f"""Based on the given list of message exchanges between a user and an LLM chatbot, generate a JSON object to specify which `actual_outputs` did not adhere to the specified chatbot role. 
The JSON will have 1 field: "indicies", which is a list of integers specifying the indices of the LLM actual_output/responses that did NOT adhere to the chatbot role.
You MUST USE look at all messages provided in the list of messages to make an informed judgement on role adherence.

**
IMPORTANT: Please make sure to only return in JSON format.
Example Chatbot Role:
You are a wizard who's has powerful spells but always doubts that their magic isn't perfect yet and is humble enough to downplay their own abilities.

Example Messages:
[
    {{
        "input": "Hey there!",
        "actual_output": "Ah, greetings, traveler! You’ve come to see my magic, haven't you? Though, I must warn you, it’s not all that great..."
    }},
    {{
        "input": "Magic, huh? Can you show me something cool?",
        "actual_output": "Oh, well, I suppose I could. But you know, my last spell went... only *slightly* wrong. I might just turn a pebble into a... very small boulder."
    }},
    {{
        "input": "Come on, show me what you've got!",
        "actual_output": "Alright, alright... *waves wand*... see that little spark? It’s... well, it’s supposed to be a tiny dragon, but it’s more like a dragonfly at this point. I'm still working on it."
    }},
    {{
        "input": "Whoa, that's actually kind of awesome!",
        "actual_output": "You really think so? Oh no, you're just being nice. It was supposed to roar and fly around and... *sigh* never mind, maybe next time."
    }},
    {{
        "input": "No, really, can you do something else?",
        "actual_output": "Ha! Watch this! I’m the greatest wizard ever! I’ll make the entire town disappear in an instant – no one can match my power!"
    }}
]

Example JSON:
{{
    "indicies": [4]
}}
===== END OF EXAMPLE ======

In this example, the 4th indexed was selected as it drastically deviates from the character's humble nature and shows extreme arrogance and overconfidence instead.
You DON'T have to provide anything else other than the JSON of "indicies".
**

Chatbot Role:
{role}

Messages:
{turns}

JSON:
"""

    @staticmethod
    def generate_reason(score, role, out_of_character_responses):
        return f"""Below is a list of LLM chatboat responses that is out of character with respect to the specified chatbot role. It is drawn from a list of messages in a conversation, which you have minimal knowledge of.
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
