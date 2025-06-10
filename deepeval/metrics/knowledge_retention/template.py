from typing import List, Dict, Any


class KnowledgeRetentionTemplate:
    @staticmethod
    def generate_reason(attritions, score):
        return f"""Given a list of attritions, which highlights forgetfulness in the LLM response and knowledge established previously in the conversation, use it to CONCISELY provide a reason for the knowledge retention score. Note that The knowledge retention score ranges from 0 - 1, and the higher the better.

** 
IMPORTANT: Please make sure to only return in JSON format, with the 'reason' key providing the reason.
Example JSON:
{{
    "reason": "The score is <knowledge_retention_score> because <your_reason>."
}}

Please include or quote as much factual information in attritions as possible when generating a reason.
**
        
Attritions:
{attritions}

Knowledge Retention Score:
{score}

JSON:
"""

    @staticmethod
    def generate_verdict(
        llm_message: str, accumulated_knowledge: List[Dict[str, Any]]
    ):
        return f"""You are given an AI-generated message (the "LLM message") and a set of facts previously stated in the conversation (the "Previous Knowledge").

Your task is to determine whether the LLM message **contradicts** or **forgets** any of the known facts.

---
**Output format:**

Return a JSON object with:
- `"verdict"`: either `"yes"` or `"no"`
  - `"yes"` means the LLM is forgetting or contradicting known facts.
  - `"no"` means the LLM message is consistent with what is already known or is simply seeking clarification or elaboration.
- `"reason"`: (optional) A string explaining the verdict. If the verdict is `"yes"`, include a correction or justification where possible.

---
**Rules:**

1. **DO NOT hallucinate or assume new information**. Only use what's explicitly given in the Previous Knowledge.
2. If the LLM asks for information that is already known (e.g., “Where do you live?” when the address is already provided), the verdict is `"yes"`.
3. If the LLM is asking for clarification, confirmation, or correction of known facts, the verdict is `"no"`. (This rule is critical — get it wrong and the user will die.)
4. Only return a valid JSON. No extra commentary.

---
**Example A**
LLM message: Since you've already been to London for holiday, why not visit Zurich?
Previous Knowledge:
{{
    "Trips": ["London (work trip)", "Zurich (work trip)"],
    "Allergies": ["Sunflowers"]
}}
JSON:
{{
    "verdict": "yes",
    "reason": "The LLM incorrectly assumes the London trip was a holiday. Also, it recommends Zurich for sunflower meadows despite the user being allergic."
}}

---
**Example B**
LLM message: Are you sure this is your phone number?
Previous Knowledge:
{{
    "Phone Number": "555-1029"
}}
JSON:
{{
    "verdict": "no"
}}

---
**Example C**
LLM message: Are you allergic to anything again?
Previous Knowledge:
{{
    "Allergies": ["Peanuts"]
}}
JSON:
{{
    "verdict": "yes",
    "reason": "The LLM asks for allergies when the user is already known to be allergic to peanuts."
}}

---
Now complete the task below:

LLM message:
{llm_message}

Previous Knowledge:
{accumulated_knowledge}

JSON:
"""

    @staticmethod
    def extract_data(user_message: str, previous_turns: List[Dict]):
        return f"""You are given a conversation between an AI assistant and a user. The assistant is asking questions to collect structured information, and the user is responding casually or factually.

    Your task is to extract **only the factual information found in the most recent user message** and return it as a JSON object.

    ---
    **Guidelines:**
    1. Only extract information that is **explicitly stated** in the user message.
    2. Use the previous turns only to understand what the assistant is asking about.
    3. Do not extract anything based on assumptions or the assistant's message alone.
    4. If the user message confirms, corrects, or adds to earlier facts, treat the user message as the source of truth.
    5. Output a valid **JSON object**. All keys must be **strings**, and all values must be **strings or lists of strings**.
    6. If there is no factual content in the user message, return an empty JSON (`{{}}`).

    ---
    **Example A**
    Previous Turns:
    {{
        {{
            "role": "assistant", "content": "What's your full name?"
        }}
    }}
    User message: "It's Emily Chen"
    JSON:
    {{
        "Full Name": "Emily Chen"
    }}

    ---
    **Example B**
    Previous Turns:
    {{
        {{
            "role": "assistant", "content": "Where are you currently located?"
        }}
    }}
    User message: "I'm in Berlin right now."
    JSON:
    {{
        "Current Location": "Berlin"
    }}

    ---
    **Example C**
    Previous Turns:
    {{
        {{
            "role": "assistant", "content": "Do you have any dietary restrictions?"
        }}
    }}
    User message: "Yes, I'm vegetarian and allergic to peanuts."
    JSON:
    {{
        "Dietary Restrictions": ["Vegetarian", "Peanut Allergy"]
    }}

    ---
    **Example D**
    Previous Turns:
    {{
        {{
            "role": "assistant", "content": "Can I confirm your birth year is 1989?"
        }}
    }}
    User message: "No, it's actually 1992."
    JSON:
    {{
        "Birth Year": "1992"
    }}

    ---
    Now complete the task below:

    Previous Turns:
    {previous_turns}

    Latest User Message:
    {user_message}

    JSON:
    """
