class PromptSynthesizerTemplate:
    @staticmethod
    def generate_synthetic_prompts(
        scenario: str, task: str, input_format: str, num_goldens: int
    ):
        return f"""Generate a series of input prompts from scratch based on the provided scenario, task, and output format.
        The inputs must align with the given scenario and task description, and conform to specified output format.

        **
        IMPORTANT: Please make sure to only return in JSON format, with the 'data' key as a list of JSON objects.
        You MUST TRY to generate {num_goldens} data points.

        Example scenario: technical SWE typing SQL queries to query from a database called FAST_FOOD_RESTAURANTS
        Example task: Text2SQL LLM Assistant
        Example input format: SQL String
        Example num prompts: 2
        Example JSON:
        {{
            "data": [
                {{
                    "input": "SELECT * FROM menu"
                }},
                {{
                    "input": "SELECT AVG(price) FROM menu;"
                }}
            ]  
        }}

        You MUST include at least one statement as the input. `input` MUST be of `{input_format}` format.
        You MUST TRY to generate {num_goldens} data points, unless the generated `input` is getting repetitive.
        **

        scenario: {scenario}
        task: {task}
        input format: {input_format}
        num prompts: {num_goldens}
        JSON:
        """

    @staticmethod
    def generate_synthetic_conversational_scenarios(
        scenario: str,
        conversational_task: str,
        participant_roles: str,
        num_goldens: int,
    ):
        return f"""
        Generate a series of conversational SCENARIOS from scratch based on the provided scenario description,
        conversational task, and participant roles.

        A SCENARIO is a narrative description of a situation in which a conversation naturally occurs.
        It is NOT a question, NOT a prompt, and NOT a user query. It MUST purely describe context.

        Each scenario MUST depict a realistic MULTI-TURN conversational situation involving the given participants.

        **
        IMPORTANT FORMAT:
        - Only return JSON
        - JSON MUST contain: {{ "data": [ {{ "scenario": "..." }}, ... ] }}
        - You MUST TRY to generate {num_goldens} items
        **

        Example of GOOD scenarios (situational descriptions):
        - "During a late afternoon code review session, a junior engineer asks their senior engineer why an async function is inconsistent, leading to a detailed back-and-forth about race conditions."
        - "While preparing for a sprint demo, a senior engineer helps a junior engineer interpret stack traces, prompting a step-by-step explanation."

        Example of BAD scenarios (DO NOT DO):
        - "Why does my async function return inconsistent results?" (This is a prompt)
        - "Explain how to debug race conditions." (Instruction)
        - "What is the freezing point of water?" (Question)

        CRITICAL REQUIREMENTS:
        - Scenario MUST be a narrative description of a SITUATION.
        - Scenario MUST involve these participant roles: {participant_roles}
        - Scenario MUST align with this conversational task: {conversational_task}
        - Scenario MUST feel natural, real-world, and MULTI-TURN.
        - Scenario MUST NOT contain:
            • direct questions
            • instructions
            • tasks
            • explicit prompts
            • standalone facts
        - Scenario MUST be grounded in the meaning of the provided base scenario description.

        You MUST TRY to generate {num_goldens} high-quality, non-repetitive scenarios.
        **

        Base Scenario Description:
        {scenario}

        Conversational Task:
        {conversational_task}

        Participant Roles:
        {participant_roles}

        Num Scenarios:
        {num_goldens}

        JSON:
        """


######################################################################################################
##### Approach similar to https://github.com/nlpxucan/WizardLM/blob/main/Evol_Instruct/depth.py ######
######################################################################################################

# generate_deepen_prompt
# "If #The Given Prompt# contains inquiries about certain issues, the depth and breadth of the inquiry can be increased."


class PromptEvolutionTemplate:

    base_instruction = """I want you to act as an input rewriter.
    Your object is the rewrite a given `input`. You MUST complicate the given `Input` using the following method:"""

    @staticmethod
    def reasoning_evolution(input):
        return (
            PromptEvolutionTemplate.base_instruction
            + f"""
            1. If `Input` can be solved with just a few simple thinking processes, you can rewrite it to explicitly request multiple-step reasoning.
            2. `Rewritten Input` should require readers to make multiple logical connections or inferences.
            3. `Rewritten Input` should be concise and understandable by humans.
            4. `Rewritten Input` should not contain more than 15 words. Use abbreviation wherever possible.

            **
            EXAMPLES

            Example input:
            Why are plants green?
            Example rewritten input:
            How does chlorophyll's role in absorbing light relate to plants' green color and their ability to produce glucose?
        
            --------------------------
            
            Example input:
            What causes seasons to change?
            Example rewritten input: 
            Given the trapping of solar radiation by atmospheric gases, explain how the enhanced activity impact Earth's climate.

            --------------------------

            Example input:
            Identify the primary factors that determine the price of goods in a market.
            Example rewritten input:
            Examine how the interplay of market demand, supply dynamics, and government policy interventions collectively shape the pricing mechanism of goods within a market ecosystem.
            **

            Input:
            {input}
            Rewritten Input:            
            """
        )

    @staticmethod
    def concretizing_evolution(input):
        return (
            PromptEvolutionTemplate.base_instruction
            + f"""
            1. Rewrite `Input` by replacing general concepts/inquiries with more specific ones.
            2. `Rewritten Input` should be concise and understandable by humans.
            3. `Rewritten Input` should not contain more than 15 words. Use abbreviation wherever possible.

            **
            EXAMPLES

            Example input: 
            Why is the biodiversity of rainforests important?
            Example rewritten input:
            How does the extensive biodiversity found in rainforests, encompassing over half of the world's plant and animal species, contribute to global biodiversity maintenance, and what role does this diversity play in enhancing ecosystem resilience, human health through disease control, crop pollination, and the development of medicines derived from rainforest plants?

            --------------------------

            Example input: 
            What is the role of bees in ecosystems?
            Example rewritten input:
            How do bees, through their pollination of flowering plants, including a multitude of fruits and vegetables, significantly influence the diversity of plant life and agricultural productivity, and in what ways do their activities extend beyond agricultural settings to support the growth of trees, flowers, and other plants, thereby providing essential resources for various animal species and contributing to the overall balance and sustainability of ecosystems?

            --------------------------

            Example input: 
            What are the principles behind solar power generation?
            Example rewritten input:
            How do photovoltaic cells work to convert sunlight into electrical power, and what role do solar panels play in this process, including energy storage for sustainable use?
            **

            Input:
            {input}
            Rewritten Input:
            """
        )

    @staticmethod
    def constrained_evolution(input):
        return (
            PromptEvolutionTemplate.base_instruction
            + f"""
            1. Rewrite `Input` by adding at least one more constraints/requirements.
            2. `Rewritten Input` should not contain more than 15 words. Use abbreviation wherever possible.

            **
            EXAMPLES

            Example input: 
            Why is the biodiversity of rainforests important?
            Example rewritten input:
            How does the biodiversity of rainforests contribute to ecosystem resilience and recovery from disturbances, and in what ways does it impact human well-being through services such as air and water purification, disease control, and crop pollination?

            --------------------------

            Example input: 
            What is the role of bees in ecosystems?
            Example rewritten input:
            Considering the pivotal role bees play in pollinating both agricultural crops and wild plants, thereby contributing to the diversity of plant life and supporting the foundation of food chains, analyze how bees influence the growth and sustainability of various ecosystems.

            --------------------------

            Example input: 
            What are the principles behind solar power generation?
            Example rewritten input:
            Examine the significance of rainforest biodiversity in sustaining ecosystem resilience and providing essential services such as disease control and crop pollination, alongside its critical role in medical research and the development of new medicines. Consider the broader implications of biodiversity loss on global ecological balance and human health.
            **

            Input:
            {input}
            Rewritten Input:
            """
        )

    @staticmethod
    def comparative_question_evolution(input):
        return (
            PromptEvolutionTemplate.base_instruction
            + f"""
            1. Rewrite `Input` to focus on comparing two or more entities, concepts, or processes.
            2. `Rewritten Input` should encourage a detailed comparison that highlights similarities and differences.
            3. `Rewritten Input` should be concise and understandable by humans.
            4. `Rewritten Input` should not contain more than 15 words. Use abbreviation wherever possible.

            **
            EXAMPLES
            
            Example input: 
            What happens to water as it boils?
            Example rewritten input:
            How does the boiling point of water at sea level compare to that of alcohol, and how does altitude affect water's boiling point?

            --------------------------

            Example input: 
            How do plants and animals process energy?
            Example rewritten input:
            Compare the processes of photosynthesis in plants and cellular respiration in animals, focusing on inputs and outputs of each process.

            --------------------------

            Example input: 
            What was the Renaissance?
            Example rewritten input:
            Contrast the main focuses and impacts of the Renaissance and the Enlightenment on European thought and culture.

            --------------------------

            Input:
            {input}
            Rewritten Input:
            """
        )

    @staticmethod
    def hypothetical_scenario_evolution(input):
        return (
            PromptEvolutionTemplate.base_instruction
            + f"""
            1. Rewrite `Input` to include a hypothetical or speculative scenario.
            2. `Rewritten Input` should encourage the reader to apply knowledge to imagine or deduce outcomes.
            3. `Rewritten Input` should be concise, clear, and understandable by humans.
            6. `Rewritten Input` should not contain more than 15 words. Use abbreviation wherever possible.

            **
            EXAMPLES

            Example input:
            What are the consequences of the greenhouse effect?
            Example rewritten input:
            Imagine a world where greenhouse gas emissions were doubled overnight. How might this intensified greenhouse effect impact global climate patterns and ecosystems?

            --------------------------

            Example input:
            How do antibiotics work?
            Example rewritten input:
            In a scenario where a new antibiotic-resistant superbug emerges, how would the principles of antibiotic action and resistance influence our approach to treatment?

            --------------------------

            Example input:
            What is quantum computing?
            Example rewritten input:
            Suppose a quantum computer was tasked with solving a problem that currently takes traditional computers centuries to solve. How might the unique capabilities of quantum computing change the outcome?
            **

            Input:
            {input}
            Rewritten Input:
            """
        )

    @staticmethod
    def in_breadth_evolution(input):
        return (
            PromptEvolutionTemplate.base_instruction
            + f"""
            1. Rewrite `Input` to create a create a brand new prompt.
            2. `Rewritten Input` should belong to the same domain as the `input` but be even more rare.
            3. `Rewritten Input` should be concise, clear, and understandable by humans.
            5. `Rewritten Input` should not contain more than 15 words. Use abbreviation wherever possible.

            **
            EXAMPLES

            Example input:
            Explore the impact of wearable technology on personal health management.
            Example rewritten input:
            Delve into the development of implantable health devices and their potential to transform chronic disease management.

            --------------------------

            Example input:
            How is quantum computing different from traditional computing?
            Example rewritten input:
            Explore the potential of quantum cryptography in enhancing cybersecurity measures beyond current encryption standards

            --------------------------

            Example input:
            What impact does virtual reality (VR) have on education?
            Example rewritten input:
            Investigate the use of VR simulations in medical training to enhance practical skills and decision-making under pressure.
            **

            Input:
            {input}
            Rewritten Input:
            """
        )


class ConversationalPromptEvolutionTemplate:

    base_instruction = """I want you to act as a conversational scenario rewriter.
    Your objective is to rewrite the given `Scenario`. You MUST complicate the `Scenario` using the following method:"""

    @staticmethod
    def reasoning_evolution(scenario):
        return (
            ConversationalPromptEvolutionTemplate.base_instruction
            + f"""
            1. Rewrite `Scenario` to force participants into multi-step conversational reasoning.
            2. Add layered inferences or analytical leaps required in dialogue.
            3. `Rewritten Scenario` must stay concise, human-readable, and remain a conversation setup.
            4. Do NOT exceed **15 words**.

            **
            EXAMPLES

            Example scenario:
            Two students discuss climate change.
            Example rewritten scenario:
            Two students debate climate impacts, tracing cause-effect chains across multiple evidence sources.

            --------------------------

            Example scenario:
            A doctor explains treatment options.
            Example rewritten scenario:
            Doctor and patient reason through symptoms requiring sequential diagnostic logic.

            --------------------------

            Scenario:
            {scenario}
            Rewritten Scenario:
            """
        )

    @staticmethod
    def concretizing_evolution(scenario):
        return (
            ConversationalPromptEvolutionTemplate.base_instruction
            + f"""
            1. Replace broad conversation setup with a **more specific, concrete** conversational scene.
            2. Add real-world detail (location, constraint, specific topic).
            3. Keep under **15 words**, concise, and still a dialogue setup.

            **
            EXAMPLES

            Example scenario:
            Two engineers talk about safety.
            Example rewritten scenario:
            Two engineers argue over failing brake-system logs during late-night review.

            --------------------------

            Example scenario:
            Two friends discuss exercise.
            Example rewritten scenario:
            Two friends compare heart-rate sensor issues during a marathon-training chat.

            --------------------------

            Scenario:
            {scenario}
            Rewritten Scenario:
            """
        )

    @staticmethod
    def constrained_evolution(scenario):
        return (
            ConversationalPromptEvolutionTemplate.base_instruction
            + f"""
            1. Add at least one new constraint shaping the conversation.
            2. Constraint must significantly affect the dialogue.
            3. Keep under **15 words**, concise, conversational.

            **
            EXAMPLES

            Example scenario:
            Two coworkers plan a report.
            Example rewritten scenario:
            Two coworkers plan a report with strict no-internet constraint.

            --------------------------

            Example scenario:
            A teacher reviews homework.
            Example rewritten scenario:
            Teacher and student discuss homework under urgent submission deadline.

            --------------------------

            Scenario:
            {scenario}
            Rewritten Scenario:
            """
        )

    @staticmethod
    def comparative_question_evolution(scenario):
        return (
            ConversationalPromptEvolutionTemplate.base_instruction
            + f"""
            1. Rewrite `Scenario` so the conversation centers on comparing two+ items.
            2. Must highlight similarities/differences through dialogue.
            3. Keep under **15 words**, concise, conversational.

            **
            EXAMPLES

            Example scenario:
            Two analysts discuss tools.
            Example rewritten scenario:
            Two analysts compare legacy analytics pipeline vs. new automated system.

            --------------------------

            Example scenario:
            Two students study history.
            Example rewritten scenario:
            Two students contrast Renaissance ideals with Enlightenment philosophies.

            --------------------------

            Scenario:
            {scenario}
            Rewritten Scenario:
            """
        )

    @staticmethod
    def hypothetical_scenario_evolution(scenario):
        return (
            ConversationalPromptEvolutionTemplate.base_instruction
            + f"""
            1. Rewrite `Scenario` to introduce a hypothetical twist derived from the setup.
            2. The hypothetical MUST drive the conversation.
            3. Keep under **15 words**, concise, conversational.

            **
            EXAMPLES

            Example scenario:
            Two scientists discuss pollution.
            Example rewritten scenario:
            Two scientists debate effects if emissions doubled overnight.

            --------------------------

            Example scenario:
            A medic trains a recruit.
            Example rewritten scenario:
            Medic and recruit plan response to hypothetical antibiotic-resistant outbreak.

            --------------------------

            Scenario:
            {scenario}
            Rewritten Scenario:
            """
        )

    @staticmethod
    def in_breadth_evolution(scenario):
        return (
            ConversationalPromptEvolutionTemplate.base_instruction
            + f"""
            1. Rewrite `Scenario` into a new conversation within the same domain.
            2. The new conversation must explore a rarer, niche angle.
            3. Keep under **15 words**, concise, conversational.

            **
            EXAMPLES

            Example scenario:
            Two doctors discuss patient care.
            Example rewritten scenario:
            Two doctors debate rare autoimmune disorder diagnostics.

            --------------------------

            Example scenario:
            Two programmers discuss bugs.
            Example rewritten scenario:
            Two programmers examine obscure concurrency race-condition failures.

            --------------------------

            Scenario:
            {scenario}
            Rewritten Scenario:
            """
        )
