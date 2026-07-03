import textwrap


class SimulatorControllerTemplate:
    @staticmethod
    def check_expected_outcome(
        previous_conversation: str, expected_outcome: str
    ) -> str:
        prompt = textwrap.dedent(
            f"""You are a Conversation Completion Checker.
            Your task is to determine whether the conversation has achieved the expected outcome and should be terminated.

            Guidelines:
            1. Review the entire conversation and decide if the expected outcome has been met and the conversation has ended.
            2. If the expected outcome has been met, mark the conversation as complete.
            3. If not, mark it as incomplete and briefly describe what remains to be done.

            IMPORTANT: The output must be formatted as a JSON object with two keys:
            `is_complete` (a boolean) and `reason` (a string).

            Example Expected Outcome: "The user has succesfully reset their password."
            Example Conversation History:
            [
                {{"role": "user", "content": "I forgot my password and need to reset it."}},
                {{"role": "assistant", "content": "Sure. First, go to the login page and click 'Forgot Password'."}},
            ]
            Example JSON Output:
            {{
                "is_complete": false,
                "reason": "The assistant explained how to forget password but ahas not confirmed that the user successfully set a new password."
            }}

            Expected Outcome: "{expected_outcome}"
            Conversation History:
            {previous_conversation}
            JSON Output:
            """
        )
        return prompt
