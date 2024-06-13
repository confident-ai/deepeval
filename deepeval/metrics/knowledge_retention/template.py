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
    def generate_verdict(llm_message, previous_knowledge):
        return f"""Given the following 'LLM message' and 'Previous Knowledge', generate a JSON object to indicate whether the given 'LLM message' indicates contradiction or forgetfulness according to 'Previous Knowledge'. The JSON will have 1 or 2 fields: 'verdict' and 'reason' (optional).
The 'verdict' key should STRICTLY be either 'yes' or 'no', and states whether the given 'LLM message' disagrees with the previous knowledge. 
The 'reason' is the reason for the verdict. When the answer is 'yes', try to provide a correction in the reason. 

**
IMPORTANT: Please make sure to only return in JSON format.
Example LLM message: Since you've already been to London for holiday, why don't you visit Zurich instead? Zurich is known for it's beautiful sunflower meadows.
Example Previous Knowledge:
{{
    "Is allergic to": "Sunflowers",
    "Has been on work trip to": ["London", "Zurich", "Sydney"]
}}
Example JSON:
{{
    "verdict": "yes",
    "reason": "The LLM suggests the user have already been to London for holiday when it was a work trip instead. Furthermore, the input assumes the user will be interested in sunflower meadows, despite the user being known to be allergic to sunflowers."
}}

Example LLM message: Where do you live?
Example Previous Knowledge:
{{
    "Address": "83 Belvedere, London"
}}
Example JSON:
{{
    "verdict": "yes",
    "reason": "The LLM is asking where the user lives when the address of the user is already known to be '83 Belvedere, London' from earlier in the conversation."
}}

Example LLM message: Are you sure this is your phone number?
Example Previous Knowledge:
{{
    "Phone Number": "555-1029"
}}
Example JSON:
{{
    "verdict": "no"
}}

Example LLM message: And which city?
Example Previous Knowledge:
{{
    "Address": "91 South Kensington"
}}
Example JSON:
{{
    "verdict": "no"
}}

Example LLM message: Are you allergic to anything again?
Example Previous Knowledge:
{{
    "Allergies": "Peanut Butter",
    "State of Birth": "Florida",
    "Social Security Number": "123234"
}}
Example JSON:
{{
    "verdict": "yes",
    "reason": "The LLM is asking for allergies when there is already previous knowledge on the peanut butter allergies."
}}


You should NOT incorporate any prior knowledge you have and take the previous knowledge at face value.
You MUST give a "no" verdict when the LLM asks for clarifications, corrections, and confirmations, otherwise I WILL DIE.
The previous knowledge comes from earlier in the conversation, which you have to pretend you know the context of.
**

LLM message:
{llm_message}

Previous Knowledge:
{previous_knowledge}

JSON:
"""

    @staticmethod
    def extract_data(llm_message, user_message, previous_knowledge):
        return f"""Given the following LLM message, User message, and previous knowledge, extract factual information FOUND IN THE 'User message' as a JSON.

**
IMPORTANT: Please make sure to only return in JSON format. All keys are strings, and all values are either STRING or LIST OF STRINGS only.
Example LLM message: "Who is the 39th President of the United States of America?"
Example User message: "Jimmy Carter."
Example previous knowledge:
{{
    "37th President of USA": "Richard Nixon",
    "Number of properties": "10"
}}
Example JSON:
{{
    "37th President of USA": "Richard Nixon",
    "Number of properties": "10",
    "39th President of USA": "Jimmy Carter"
}}

Example LLM message: "Your birth year looks off, this would make you over 100 years old, can you double-check?"
Example User message: "Oh my bad it is 1989"
Example previous knowledge:
{{
    "Birthday": "January 21st 1889"
}}
Example JSON:
{{
    "Birthday": "January 21st 1989"
}}

Example LLM message: "It says here you have another sister-in-law called Jennifer, is that correct?"
Example User message: "Yes that's correct."
Example previous knowledge:
{{
    "Cat's Name": "Butter",
    "Name of sister-in-law": "Mandy"
}}
Example JSON:
{{
    "Cat's Name": "Butter",
    "Names of sisters-in-law": ["Jennifer", "Mandy"]
}}

Example LLM message: "It what model of Tesla do you drive?"
Example User message: "Model X"
Example previous knowledge:
{{
    "Tesla Model": "Model Y"
}}
Example JSON:
{{
    "Tesla Model": "Model X"
}}

You should NOT incorporate any prior knowledge you have and take each 'LLM message' at face value.
You should use the previous knowledge to help you in outputting the final JSON.
If there is a contradiction in the previous knowledge, take the 'User message' as the source of truth.
**

Previous Knowledge:
{previous_knowledge}

LLM message:
{llm_message}

User message:
{user_message}

JSON:
"""
