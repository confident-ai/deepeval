from typing import List, Dict


class ConversationalGEvalTemplate:
    @staticmethod
    def generate_evaluation_steps(parameters: str, criteria: str):
        return f"""Given an evaluation criteria which outlines how you should judge a conversation between a user and an LLM chatbot using the {parameters} fields in each turn, generate 3-4 concise evaluation steps based on the criteria below. Based on the evaluation criteria, you MUST make it clear how to evaluate the {parameters} in relation to one another in each turn, as well as the overall quality of the conversation.

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
    def generate_evaluation_results(
        evaluation_steps: str, turns: List[Dict], parameters: str
    ) -> str:
        return f"""You are given a set of **Evaluation Steps** that describe how to assess a conversation between a user and an LLM chatbot. Your task is to return a JSON object with exactly two fields:

1. `"score"`: An integer from 0 to 10 (inclusive), where:
   - 10 = The conversation *fully* meets the criteria described in the Evaluation Steps.
   - 0 = The conversation *completely fails* to meet the criteria.

2. `"reason"`: A **concise but precise** explanation for the score. Your reasoning **must reference specific aspects of the Evaluation Steps** and **mention relevant details from the conversation and the given parameters**. DO NOT include the score value in your explanation.

Evaluation Steps:
{evaluation_steps}

Conversation:
{turns}

Parameters to consider during evaluation:
{parameters}

**
IMPORTANT: You MUST return only a valid JSON object with the exact keys "score" and "reason". No additional text, commentary, or formatting.
**

Example JSON:
{{
  "score": 0,
  "reason": "your concise and informative reason here"
}}
"""
