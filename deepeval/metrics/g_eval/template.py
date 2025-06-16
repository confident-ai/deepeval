from typing import List, Optional, Tuple
from deepeval.metrics.g_eval.utils import Rubric


class GEvalTemplate:
    @staticmethod
    def generate_evaluation_steps(parameters: str, criteria: str):
        return f"""Given an evaluation criteria which outlines how you should judge the {parameters}, generate 3-4 concise evaluation steps based on the criteria below. You MUST make it clear how to evaluate {parameters} in relation to one another.

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
        test_case_content: str,
        parameters: str,
        rubric: Optional[str] = None,
        score_range: Tuple[int, int] = (0, 10),
        _additional_context: Optional[str] = None,
    ):
        rubric_text = f"Rubric:\n{rubric}\n" if rubric else ""
        dependencies = (
            "evaluation steps and rubric" if rubric else "evaluation steps"
        )
        score_explanation = (
            "based on the rubric provided"
            if rubric
            else f"with {score_range[1]} indicating strong alignment with the evaluation steps and {score_range[0]} indicating no alignment"
        )
        reasoning_expectation = (
            "Be specific and grounded in the evaluation steps and rubric."
            if rubric
            else "Be specific and grounded in the evaluation steps."
        )
        additional_context = (
            f"\n\nAdditional Context:\n{_additional_context}\n"
            if _additional_context
            else ""
        )

        return f"""You are an evaluator. Given the following {dependencies}, assess the response below and return a JSON object with two fields:

    - `"score"`: an integer between {score_range[0]} and {score_range[1]}, {score_explanation}.
    - `"reason"`: a brief explanation for why the score was given. This must mention specific strengths or shortcomings, referencing relevant details from the input. Do **not** quote the score itself in the explanation.

    Your explanation should:
    - {reasoning_expectation}
    - Mention key details from the test case parameters.
    - Be concise, clear, and focused on the evaluation logic.

    Only return valid JSON. Do **not** include any extra commentary or text.

    ---

    Evaluation Steps:
    {evaluation_steps}

    {rubric_text}
    Test Case:
    {test_case_content}

    Parameters:
    {parameters}
    {additional_context}

    ---
    **Example JSON:**
    {{
        "score": {score_range[0]},
        "reason": "your concise and informative reason here"
    }}

    JSON:
    """

    @staticmethod
    def generate_strict_evaluation_results(
        evaluation_steps: str,
        test_case_content: str,
        parameters: str,
        _additional_context: Optional[str] = None,
    ):
        additional_context = (
            f"\n\nAdditional Context:\n{_additional_context}\n"
            if _additional_context
            else ""
        )
        return f"""Given the evaluation steps, return a JSON with two keys: 1) a `score` key that is STRICTLY EITHER 1 (follows the criteria 100% outlined in the evaluation steps), OR 0 (does not follow the criteria), and 2) a `reason` key, a reason for the given score, but DO NOT QUOTE THE SCORE in your reason. Please mention specific information from {parameters} in your reason, but be very concise with it!

Evaluation Steps:
{evaluation_steps}

{test_case_content}
{additional_context}
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
