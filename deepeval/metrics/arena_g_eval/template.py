from typing import Dict, List, Optional
import textwrap


class ArenaGEvalTemplate:
    @staticmethod
    def generate_evaluation_steps(parameters: str, criteria: str):
        return textwrap.dedent(
            f"""Given an evaluation criteria which outlines how you should choose the winner out of all contestants based on the {parameters}, generate 3-4 concise evaluation steps based on the criteria below. You MUST make it clear how to evaluate {parameters} in relation to one another.

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
        )

    @staticmethod
    def generate_arena_winner(
        evaluation_steps: str,
        test_case_contents: List[str],
        parameters: str,
    ):
        reasoning_expectation = (
            "Be specific and grounded in the evaluation steps."
        )

        return textwrap.dedent(
            f"""
            You are a judge. Given the following evaluation steps, select the single contestant that best aligns with the evaluation steps.
            Return a JSON object with three fields:

            - `"winner"`: the contestant that is best aligned with the evaluation steps.
            - `"reason"`: a brief explanation for why the contestant was chosen. This must mention specific strengths or shortcomings, and reference relevant details from BOTH the winner's parameters AND ALL the other contestants' parameters, but DO NOT mention the contestant indeces, and refer to the contestants ONLY by their Contestant Name formatted by wrapping in $contesntant_name$.

            Your explanation should:
            - {reasoning_expectation}
            - Mention key details from the contestants' parameters.
            - Be concise, clear, and focused on the evaluation logic.
            - Wrap the contestant name in $contesntant_name$ when referring to the contestant.

            !!! IMPORTANT 
            Refer to contestants ONLY by their unique contestant name.
            !!! 

            Only return valid JSON. Do **not** include any extra commentary or text.

            ---

            Evaluation Steps:
            {evaluation_steps}

            Contestants:
            {test_case_contents}

            Parameters:
            {parameters}

            ---
            **Example JSON:**
            {{
                "winner": <contestant>,
                "reason": <your-concise-and-informative-reason-here>
            }}

            JSON:
        """
        )

    @staticmethod
    def rewrite_reason(
        reason: str,
        dummy_to_real_names: Dict[str, str],
    ):
        return textwrap.dedent(
            f"""
            Given the following reason that explains which contestant is the winner, rewrite the reason to REPLACE all contestant names with their real names.

            The contestant names are wrapped in $name$ format (e.g., $Alice$, $Bob$, $Charlie$).
            
            Use the provided dummy-to-real names mapping to convert each $dummy_name$ to its corresponding real name.

            Dummy-to-real mapping:
            {dummy_to_real_names}

            Reason:
            {reason}

            **Instructions:**
            1. Find all instances of $name$ in the reason text
            2. Look up each name in the dummy_to_real_names mapping
            3. Replace $name$ with the corresponding real name
            4. Keep all other text unchanged

            **Example:**
            If mapping is {{"Alice": "gpt-4", "Bob": "claude-3"}} and reason contains "$Alice$ provided better answers than $Bob$", 
            the result should be "gpt-4 provided better answers than claude-3"

            Return only the rewritten reason as JSON.

            ---
            **Example JSON:**
            {{
                "rewritten_reason": <your-rewritten-reason-here>
            }}

            JSON:
            """
        )
