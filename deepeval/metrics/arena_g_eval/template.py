from typing import List, Optional
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
            - `"reason"`: a brief explanation for why the contestant was chosen. This must mention specific strengths or shortcomings, and reference relevant details from BOTH the winner's parameters AND ALL the other contestants' parameters, but DO NOT mention the contestant indeces, and refer to the contestants ONLY by their Contestant ID.

            Your explanation should:
            - {reasoning_expectation}
            - Mention key details from the contestants' parameters.
            - Be concise, clear, and focused on the evaluation logic.

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
