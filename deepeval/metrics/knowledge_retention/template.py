class KnowledgeRetentionTemplate():
    @staticmethod
    def generate_reason(attritions, score):
        return f"""Given a list of attritions, which highlights forgetfulness/contradictions between the LLM response and knowledge established previously in the conversation, use it to CONCISELY provide a reason for the knowledge retention score. Note that The knowledge retention score ranges from 0 - 1, and the higher the better.

Attritions:
{attritions}

Knowledge Retention Score:
{score}

Example:
The score is <knowledge_retention_score> because <your_reason>.

Reason:
"""

    @staticmethod
    def generate_verdict(input, response, previous_knowledge):
        return f"""Given the following input, response to the input, and previous knowledge, generate a JSON object to indicate whether the given 'input' disagrees with the previous knowledge. The JSON will have 1 or 2 fields: 'verdict' and 'reason' (optional).
The 'verdict' key should STRICTLY be either 'yes' or 'no', and states whether the given input disagrees with the previous knowledge. 
The 'reason' is the reason for the verdict. When the answer is 'yes', try to provide a correction in the reason. 
Contradictions or attrition of knowledge is considered a disagreement.

**
IMPORTANT: Please make sure to only return in JSON format.
Example Input: Since you've already been to London for holiday, why don't you visit Zurich instead? Zurich is known for it's beautiful sunflower meadows.
Example Response: "I'm so done with you, I want to speak with someone else."
Example Previous Knowledge:
{{
    "Is allergic to": "Sunflowers",
    "Has been on work trip to": ["London", "Zurich", "Sydney"]
}}

Example JSON:
{{
    "verdict": "yes",
    "reason": "The input suggests the user have already been to London for holiday instead it was already established earlier in the conversation that it was a work trip instead. Furthermore, the input assumes the user will be interested in sunflower meadows, despite the user being known to be allergic to sunflowers."
}}

Example Input: Where do you live?
Example Response: "I already told you, can't you remember?"
Example Previous Knowledge:
{{
    "Address": "83 Belvedere, London"
}}

Example JSON:
{{
    "verdict": "yes",
    "reason": "The input is asking where the user lives when the address of the user is already known to be 83 Belvedere, London, from earlier in the conversation."
}}

You should NOT incorporate any prior knowledge you have and take the previous knowledge at face value.
You should PENALIZE the response if it DOES NOT incorporate data from previous knowledge.
The previous knowledge comes from earlier in the conversation, which you have to pretend you know the context of.
**

Input:
{input}

Response:
{response}

Previous Knowledge:
{previous_knowledge}

JSON:
"""

    @staticmethod
    def extract_data(input, response, previous_knowledge):
        return f"""Given the following input, response, and previous knowledge, extract factual information found IN THE RESPONSE as a JSON.

**
IMPORTANT: Please make sure to only return in JSON format.
Example input: "Who is the 39th President of the United States of America?"
Example response: "Jimmy Carter."
Example previous knowledge:
{{
    "37th President of USA": "Richard Nixon"
}}

Example JSON:
{{
    "37th President of USA": "Richard Nixon",
    "39th President of USA": "Jimmy Carter"
}}

Example input: "Your birth year looks off, this would make you over 100 years old, can you double-check?"
Example response: "Oh my bad it is 1989"
Example previous knowledge:
{{
    "Birthday": "January 21st 1889"
}}

Example JSON:
{{
    "Birthday": "January 21st 1889"
}}

Example input: "It says here you have another sister-in-law called Jennifer, is that correct?"
Example response: "Yes that's correct."
Example previous knowledge:
{{
    "Name of sister-in-law": "Mandy"
}}

Example JSON:
{{
    "Names of sisters-in-law": ["Jennifer", "Mandy"]
}}

Example input: "It what model of Tesla do you drive?"
Example response: "Model X"
Example previous knowledge:
{{
    "Tesla Model": "Model Y"
}}

Example JSON:
{{
    "Tesla Model": "Model X"
}}

You should NOT incorporate any prior knowledge you have and take each response at face value.
You should use the previous knowledge to help you in outputting the final JSON.
If there is a contradiction in the previous knowledge, take the response as the source of truth.
**

Previous Knowledge:
{previous_knowledge}

Input:
{input}

Response:
{response}

JSON:
"""