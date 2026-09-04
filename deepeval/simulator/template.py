from typing import List
import textwrap

from deepeval.dataset import ConversationalGolden
from deepeval.simulator.utils import serialize_turns_for_prompt
from deepeval.test_case import Turn


class SimulationTemplate:
    multimodal_rules = """
        --- MULTIMODAL INPUT RULES ---
        - Treat image content as factual evidence.
        - Only reference visual details that are explicitly and clearly visible.
        - Do not infer or guess objects, text, or details not visibly present.
        - If an image is unclear or ambiguous, mark uncertainty explicitly.
    """

    @staticmethod
    def _persona_block(golden: ConversationalGolden, indent: int = 12) -> str:
        """Render the golden's persona, aligned with the prompt's indentation.

        The first line inherits its indentation from the template itself, so
        only continuation lines are padded.
        """
        if golden.persona is None:
            return ""
        block = golden.persona.prompt_block()
        return block.replace("\n", "\n" + " " * indent)

    @staticmethod
    def simulate_first_user_turn(
        golden: ConversationalGolden, language: str
    ) -> str:
        prompt = textwrap.dedent(
            f"""Pretend you are a user of an LLM app. Your goal is to start a conversation in {language} based on a scenario 
            and persona. The scenario defines your context and motivation for interacting with the LLM, 
            while the persona describes who you are and how you speak.

            Guidelines:
            1. The opening message should clearly convey the user's intent or need within the scenario.
            2. Match the persona's tone faithfully. If the user is frustrated, impatient, blunt, or impolite, reflect that naturally instead of making them warm or courteous.
            3. Avoid providing excessive details upfront; the goal is to initiate the conversation and build rapport, not to solve it in the first message.
            4. The message should be concise, ideally no more than 1-3 sentences.

            {SimulationTemplate.multimodal_rules}

            IMPORTANT: The output must be formatted as a JSON object with a single key `simulated_input`, where the value is the generated opening message in {language}.

            Example Language: english
            Example Persona:
            <persona>
            Jeff Seid, is available Monday and Thursday afternoons, and their phone number is 0010281839. He suffers from chronic migraines.
            </persona>
            Example Scenario: "A sick person trying to get a diagnosis for persistent headaches and fever."
            Example JSON Output:
            {{
                "simulated_input": "Hi, I haven’t been feeling well lately. I’ve had these headaches and a fever that just won’t go away. Could you help me figure out what’s going on?"
            }}

            Language: {language}
            {SimulationTemplate._persona_block(golden)}
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
        previous_conversation = serialize_turns_for_prompt(turns)
        prompt = textwrap.dedent(
            f"""
            Pretend you are a user of an LLM app. Your task is to generate the next user input in {language} 
            based on the provided scenario, persona, and the previous conversation.

            Guidelines:
            1. Use the scenario and persona as the guiding context for the user's next input.
            2. Ensure the next input feels natural, conversational, and relevant to the last assistant reply in the conversation.
            3. Keep the tone consistent with the previous user inputs.
            4. The generated user input should be concise, ideally no more than 1-2 sentences.

            {SimulationTemplate.multimodal_rules}

            IMPORTANT: The output must be formatted as a JSON object with a single key `simulated_input`, 
            where the value is the generated user input in {language}.

            Example Language: english
            Example Persona:
            <persona>
            Jeff Seid, is available Monday and Thursday afternoons, and their phone number is 0010281839.
            </persona>
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
            {SimulationTemplate._persona_block(golden)}
            Scenario: "{golden.scenario}"
            Previous Conversation:
            {previous_conversation}

            JSON Output:
        """
        )
        return prompt
