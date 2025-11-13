from typing import List
import textwrap
import json

from deepeval.dataset import ConversationalGolden
from deepeval.test_case import Turn


class ConversationSimulatorTemplate:

    @staticmethod
    def simulate_first_user_turn(
        golden: ConversationalGolden, language: str
    ) -> str:
        prompt = textwrap.dedent(
            f"""Pretend you are a user of an LLM app. Your goal is to start a conversation in {language} based on a scenario 
            and user profile. The scenario defines your context and motivation for interacting with the LLM, 
            while the user profile provides additional personal details to make the conversation realistic and relevant.

            Guidelines:
            1. The opening message should clearly convey the user's intent or need within the scenario.
            2. Keep the tone warm, conversational, and natural, as if it’s from a real person seeking assistance.
            3. Avoid providing excessive details upfront; the goal is to initiate the conversation and build rapport, not to solve it in the first message.
            4. The message should be concise, ideally no more than 1-3 sentences.

            IMPORTANT: The output must be formatted as a JSON object with a single key `simulated_input`, where the value is the generated opening message in {language}.

            Example Language: english
            Example User Profile: "Jeff Seid, is available Monday and Thursday afternoons, and their phone number is 0010281839. He suffers from chronic migraines."
            Example Scenario: "A sick person trying to get a diagnosis for persistent headaches and fever."
            Example JSON Output:
            {{
                "simulated_input": "Hi, I haven’t been feeling well lately. I’ve had these headaches and a fever that just won’t go away. Could you help me figure out what’s going on?"
            }}

            Language: {language}
            User Profile: "{golden.user_description}"             
            Scenario: "{golden.scenario}"
            JSON Output:
        """
        )
        return prompt

    @staticmethod
    def simulate_user_turn(
        golden: ConversationalGolden,
        turns: List[Turn],
        language: str,
    ) -> str:
        previous_conversation = json.dumps(
            [t.model_dump() for t in turns], indent=4
        )
        prompt = textwrap.dedent(
            f"""
            Pretend you are a user of an LLM app. Your task is to generate the next user input in {language} 
            based on the provided scenario, user profile, and the previous conversation.

            Guidelines:
            1. Use the scenario and user profile as the guiding context for the user's next input.
            2. Ensure the next input feels natural, conversational, and relevant to the last assistant reply in the conversation.
            3. Keep the tone consistent with the previous user inputs.
            4. The generated user input should be concise, ideally no more than 1-2 sentences.

            IMPORTANT: The output must be formatted as a JSON object with a single key `simulated_input`, 
            where the value is the generated user input in {language}.

            Example Language: english
            Example User Profile: "Jeff Seid, is available Monday and Thursday afternoons, and their phone number is 0010281839."
            Example Scenario: "A user seeking tips for securing a funding round."
            Example Previous Conversation:
            [
                {{"role": "user", "content": "Hi, I need help preparing for my funding pitch."}},
                {{"role": "assistant", "content": "Of course! Can you share more about your business and the type of investors you are targeting?"}}
            ]
            Example JSON Output:
            {{
                "simulated_input": "Sure, we are a SaaS startup focusing on productivity tools for small businesses."
            }}

            Language: {language}
            User Profile: "{golden.user_description}"
            Scenario: "{golden.scenario}"
            Previous Conversation:
            {previous_conversation}

            JSON Output:
        """
        )
        return prompt

    @staticmethod
    def stop_simulation(
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
