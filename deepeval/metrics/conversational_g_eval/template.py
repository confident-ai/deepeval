class ConversationalGEvalTemplate:
    @staticmethod
    def generate_evaluation_steps(parameters, criteria):
        return f"""Given an evaluation criteria which outlines how you should judge a conversation between a user and an LLM chatbot using the {parameters} fields in each turn, generate 3-4 concise evaluation steps based on the criteria below. Based on the evalution criteria, you MUST make it clear how to evaluate the {parameters} in relation to one another in each turn, as well as the overall quality of the conversation.

Evaluation Criteria:
{criteria}

**
IMPORTANT: Please make sure to only return in JSON format, with the "steps" key as a list of strings. No words or explanation is needed.
Example JSON:
{{
    "steps": <list_of_strings>
}}
**

JSON:
"""

    @staticmethod
    def generate_evaluation_results(evaluation_steps, conversation, parameters):
        return f"""Given the evaluation steps that outlines how to evaluate a conversation between a user and an LLM chatbot, return a JSON with two keys: 1) a `score` key ranging from 0 - 10, with 10 being that it follows the criteria outlined in the steps and 0 being that it does not, and 2) a `reason` key, a reason for the given score, but DO NOT QUOTE THE SCORE in your reason. Please mention specific information from {parameters} in the conversation in your reason, but be very concise with it!

Evaluation Steps:
{evaluation_steps}

Conversation:
{conversation}

**
IMPORTANT: Please make sure to only return in JSON format, with the "score" and "reason" key. No words or explanation is needed.

Example JSON:
{{
    "score": 0,
    "reason": "The text does not follow the evaluation steps provided."
}}
**

JSON:
"""
