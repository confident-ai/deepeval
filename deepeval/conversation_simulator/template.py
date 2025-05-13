import textwrap


class ConversationSimulatorTemplate:
    @staticmethod
    def generate_user_profile(user_profile_items: str, language: str) -> str:
        prompt = textwrap.dedent(
            f"""You are a User Profile Generator. Your task is to create a concise, natural-language user profile summary in {language}
            based on the given profile items.

            Guidelines:
            1. Use the provided items to generate a coherent, human-readable profile.
            2. Ensure the profile is concise and includes all the information described in the input requirements, seamlessly integrated into a natural format.
            3. Avoid rigidly listing the items; instead, craft the profile as if writing a short and concise bio.

            IMPORTANT: The output must be formatted as a JSON object with a single key `user_profile`, where the value 
            is the generated user profile in {language}.

            Example Language: english
            Example User Profile Items:
            ["name (first and last)", "phone number", "availabilities (between Monday and Friday)"]
            Example JSON Output:
            {{
                "user_profile": "Jeff Seid is available on Monday and Thursday afternoons, and his phone number is 0010281839."
            }}

            Language: {language}
            User Profile Items:
            "{user_profile_items}"
            JSON Output:
        """
        )
        return prompt

    @staticmethod
    def generate_scenario(user_profile: str, intent: str, language: str) -> str:
        prompt = textwrap.dedent(
            f"""You are a Scenario Generator. Your task is to create a detailed and realistic scenario in {language}
            based on the given user profile and their intent for interacting with the LLM.

            Guidelines:
            1. Use the user profile to establish the context, incorporating any relevant personal details (e.g., name, availability, location).
            2. Clearly articulate the user's intent, turning it into a concrete, real-life situation.
            3. The scenario should be written in a natural, narrative tone and provide enough background to understand the user's motivation and interaction goals.
            4. Ensure the scenario is concise but specific enough to describe the context and purpose.

            IMPORTANT: The output must be formatted as a JSON object with a single key `scenario`, where the value is the generated scenario description in {language}.

            Example Language: english
            Example User Profile: "Jeff Seid, is available Monday and Thursday afternoons, and their phone number is 0010281839."
            Example Intent: "Picking products up because somebody died or moved into intensive care."
            Example Output:
            {{
                "scenario": "Your name is Jeff Seid, and your mother Karen Springfield (born 19th of August 1949) recently passed away. You still have a wheelchair and a wheeled walker at her house at 545 Stranger Road, which should be picked up by the provider. You are currently performing a phone call with the provider to arrange the retrieval of these products."
            }}

            Language: {language}
            User Profile: "{user_profile}"
            Intent: "{intent}"
            JSON Output:
        """
        )
        return prompt

    @staticmethod
    def generate_first_input(
        scenario: str, user_profile: str, language: str
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
            User Profile: "{user_profile}"             
            Scenario: "{scenario}"
            JSON Output:
        """
        )
        return prompt

    @staticmethod
    def generate_next_user_input(
        scenario: str,
        user_profile: str,
        previous_conversation: str,
        language: str,
    ) -> str:
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
            User Profile: "{user_profile}"
            Scenario: "{scenario}"
            Previous Conversation:
            {previous_conversation}

            JSON Output:
        """
        )
        return prompt

    @staticmethod
    def check_conversation_completed(
        previous_conversation: str, stopping_criteria: str
    ) -> str:
        prompt = textwrap.dedent(
            f"""You are a Conversation Completion Checker.
            Your task is to determine whether the conversation satisfies the provided stopping criteria and should be terminated.

            Guidelines:
            1. Review the entire conversation and decide if the stopping criteria have been met and the conversation has ended.
            2. If the criteria have been met, mark the conversation as complete.
            3. If not, mark it as incomplete and briefly describe what remains to be done.

            IMPORTANT: The output must be formatted as a JSON object with two keys:
            `is_complete` (a boolean) and `reason` (a string).

            Example Stopping Criteria: "The user has succesfully reset their password."
            Example Conversation History:
            [
                {{"role": "user", "content": "I forgot my password and need to reset it."}},
                {{"role": "assistant", "content": "Sure. First, go to the login page and click 'Forgot Password'."}},
            ]
            Example JSON Output:
            {{
                "is_complete": False,
                "reason": "The assistant explained how to forget password but ahas not confirmed that the user successfully set a new password."
            }}

            Stopping Criteria: "{stopping_criteria}"
            Conversation History:
            {previous_conversation}
            JSON Output:
            """
        )
        return prompt
