from typing import List, Dict, Optional


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
        evaluation_steps: str,
        turns: List[Dict],
        parameters: str,
        rubric: Optional[str] = None,
    ) -> str:
        rubric_text = f"Rubric:\n{rubric}\n" if rubric else ""
        dependencies = (
            "Evaluation Steps and Rubric" if rubric else "Evaluation Steps"
        )
        score_explanation = (
            "based on how well the conversation follows the rubric and evaluation steps"
            if rubric
            else "based on how well the conversation follows the evaluation steps"
        )
        reasoning_guidance = (
            "Your reasoning must reference specific aspects of both the rubric and the evaluation steps,"
            if rubric
            else "Your reasoning must reference specific aspects of the evaluation steps,"
        )

        return f"""You are given a set of {dependencies} that describe how to assess a conversation between a user and an LLM chatbot. Your task is to return a JSON object with exactly two fields:

    1. `"score"`: An integer from 0 to 10 (inclusive), where:
    - 10 = The conversation *fully* meets the criteria described in the Evaluation Steps
    - 0 = The conversation *completely fails* to meet the criteria
    - All other scores represent varying degrees of partial fulfillment,
    {score_explanation}.

    2. `"reason"`: A **concise but precise** explanation for the score. {reasoning_guidance} and mention relevant details from the conversation and the given parameters. DO NOT include the score value in your explanation.

    Evaluation Steps:
    {evaluation_steps}

    {rubric_text}Conversation:
    {turns}

    Parameters to consider during evaluation:
    {parameters}

    ---
    IMPORTANT: You MUST return only a valid JSON object with the exact keys `"score"` and `"reason"`. No additional text, commentary, or formatting.

    ---
    Example JSON:
    {{
    "score": 0,
    "reason": "Your concise and informative reason here."
    }}
    """
