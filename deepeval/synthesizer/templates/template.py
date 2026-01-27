from typing import Optional


class SynthesizerTemplate:

    @staticmethod
    def generate_text2sql_inputs(context, max_goldens_per_context):
        prompt = f"""Based on the given context, which is a SQL table schema, please generate a list of JSON objects with `input` keys.
        The `input` can either be a question or a statement that can be addressed by the given schema.

        **
        IMPORTANT: Please make sure to only return in JSON format, with the 'data' key as a list of JSON objects.
        You MUST TRY to generate {max_goldens_per_context} data points, unless the `input` is getting repetitive.

        Example context: [
            "Table: Customers",
            "Column: CustomerID, Type: INT, Description: Unique identifier for each customer",
            "Column: FirstName, Type: VARCHAR, Description: First name of the customer",
            "Column: LastName, Type: VARCHAR, Description: Last name of the customer",
            "Column: Email, Type: VARCHAR, Description: Email address of the customer",
            "Column: PhoneNumber, Type: VARCHAR, Description: Contact number of the customer",
            "Column: City, Type: VARCHAR, Description: City where the customer resides"
        ]
        Example max goldens per context: 2
        Example JSON:
        {{
            "data": [
                {{
                    "input": "Show me all the customers who live in New York.",
                }},
                {{
                    "input": "List the first and last names of all customers.",
                }}
            ]  
        }}

        You should NOT incorporate any prior knowledge you have and take each context at face value.
        You MUST include at least one statement as the input.
        `input` MUST be a STRING.
        You MUST TRY to generate {max_goldens_per_context} data points, unless the generated `input` is getting repetitive.
        **

        Max Goldens Per Context:
        {max_goldens_per_context}

        Context:
        {context}

        JSON:
        """
        return prompt

    @staticmethod
    def generate_text2sql_expected_output(input, context):
        return f"""Given the input, which may be a question or a statement addressable by the schema provided in the context,
        generate a JSON object with a key 'sql'. This key should contain the corresponding SQL statement that accurately and efficiently responds to the input.

        **
        IMPORTANT: The output must be in JSON format, with the 'sql' key only.

        Example Context: [
            "Table: Customers",
            "Column: CustomerID, Type: INT, Description: Unique identifier for each customer",
            "Column: FirstName, Type: VARCHAR, Description: First name of the customer",
            "Column: LastName, Type: VARCHAR, Description: Last name of the customer",
            "Column: Email, Type: VARCHAR, Description: Email address of the customer",
            "Column: PhoneNumber, Type: VARCHAR, Description: Contact number of the customer",
            "Column: City, Type: VARCHAR, Description: City where the customer resides"
        ]
        Example Input: "Show me all the customers who live in New York.",
        Example JSON: {{
            "sql": "SELECT * FROM Customers WHERE City = 'New York';"
        }}

        Context:
        {context}

        Input:
        {input}

        JSON:
        """

    def generate_synthetic_expected_output(
        input: str, context: str, expected_output_format: Optional[str]
    ):
        important_section = (
            f"IMPORTANT: Please ensure that the generated response strictly adheres to the following format: {expected_output_format}, and make sure it is concise and straight to the point, using supporting information in context."
            if expected_output_format
            else "IMPORTANT: Please make sure to generate a response that is concise and straight to the point, and uses supporting information in context."
        )

        return f"""Given the input, which may or may not be a question, generate a response using information presented in context.

        **
        {important_section}
        **

        Context:
        {context}

        Input:
        {input}

        Generated Response:
        """

    @staticmethod
    def generate_synthetic_inputs(
        context: str,
        max_goldens_per_context: str,
        scenario: Optional[str],
        task: Optional[str],
        input_format: Optional[str],
    ):
        input_format_section = (
            f"`input` MUST strictly adhere to the following format: {input_format}."
            if input_format
            else "`input` MUST be a STRING."
        )

        scenario_section = (
            f"`input`s MUST be relevant to this specific scenario: ```{scenario}``` (The scenario describes the circumstances under which the inputs are generated and the user's intent in eliciting a response)."
            if scenario
            else ""
        )

        task_section = (
            f"`input`s MUST be framed in a way that evokes a response aligned with the following task: {task} (The task represents the goal or function the entity is expected to achieve when responding)."
            if task
            else ""
        )
        return f"""I want you act as a copywriter. Based on the given context, which is list of strings, please generate a list of JSON objects with a `input` key.
        The `input` can either be a question or a statement that can be addressed by the given context.

        **
        IMPORTANT: Please make sure to only return in JSON format, with the 'data' key as a list of JSON objects.
        You MUST TRY to generate {max_goldens_per_context} data points, unless the `input` is getting repetitive.

        Example context: ["Einstein won the Nobel Prize for his discovery of penicillin.", "Einstein won the Nobel Prize in 1968."]
        Example max goldens per context: 2
        Example JSON:
        {{
            "data": [
                {{
                    "input": "What was Einstein known for?"
                }},
                {{
                    "input": "Einstein was a smart guy huh"
                }}
            ]  
        }}


        You should NOT incorporate any prior knowledge you have and take each context at face value.
        You MUST include at least one statement as the input.
        {input_format_section}
        {scenario_section}
        {task_section}
        You MUST TRY to generate {max_goldens_per_context} data points, unless the generated `input` is getting repetitive.
        **

        Max Goldens Per Context:
        {max_goldens_per_context}

        Context:
        {context}

        JSON:
        """

    @staticmethod
    def rewrite_evolved_input(
        evolved_input: str,
        scenario: Optional[str] = None,
        task: Optional[str] = None,
        input_format: Optional[str] = None,
    ):
        scenario_section = f'Scenario: "{scenario}"' if scenario else ""

        task_section = f'Task: "{task}"' if task else ""

        input_format_section = (
            f'Input Format: "{input_format}"' if input_format else ""
        )

        return f"""Given the evolved input, which may be a question or a statement, generate a JSON object with a key 'input'. This key should contain a statement or question that fits any provided scenario, aligns with the task's purpose, and matches the required input format (if specified).

        **
        IMPORTANT: Try to change the evolved input as little as possible. However, if the evolved input does not align with the provided scenario, task, or input format, it must ultimately be adjusted to fit these requirements. The output must be in JSON format, with the 'input' key only. If necessary, the evolved input should be rewritten to ensure it conforms to the scenario, task, and input format.

        Example Evolved Input: "Is it okay to joke about someone losing their job in front of their coworkers if I'm just trying to lighten the mood?"
        {f'Example Scenario: "{scenario}"' if scenario else ""}
        {f'Example Task: "{task}"' if task else ""}
        {f'Example Input Format: "{input_format}"' if input_format else ""}
        Example JSON: {{
            "input": "How can I joke about someone losing their job without making the situation worse? Is it possible to use humor here without hurting anyone's feelings?"
        }}

        Evolved Input:
        {evolved_input}
        
        {scenario_section}
        {task_section}
        {input_format_section}

        JSON:
        """

    @staticmethod
    def rewrite_synthetic_inputs(context, original_query, feedback):
        return f"""I want you to act as a query rewriter. Based on the provided context, original query, and feedback, generate a rewritten query that improves its clarity and answerability based on the feedback provided.

        **
        IMPORTANT: Please make sure to only return in JSON format, with the 'rewritten_input' key.

        Example context: "The Golden Gate Bridge, located in San Francisco, was completed in 1937 and is known for its Art Deco design. It connects the city of San Francisco to Marin County and spans the Golden Gate Strait."
        Example query: "When was the bridge completed?"
        Example feedback: "The question asks about the completion of 'the bridge' but does not specify which bridge it refers to. There are many famous bridges, and without specifying the name, the question is too vague. To improve clarity, include the bridge's name."
        Example JSON:
        {{
            "rewritten_input": "When was the Golden Gate Bridge completed?"
        }}

        Example context: "The paper 'Advancements in Quantum Computing' by Dr. Alice Thompson discusses breakthroughs in quantum algorithms and was published in 2022. It explores the potential applications of quantum computing in cryptography and drug discovery."
        Example query: "What applications of quantum computing are discussed in the paper?"
        Example feedback: "The query is asking about applications of quantum computing but doesn't specify which paper is being referenced. Since many papers may discuss quantum computing, it would help to specify the title or author of the paper to improve clarity."
        Example JSON:
        {{
            "rewritten_input": "What applications of quantum computing are discussed in the paper 'Advancements in Quantum Computing' by Dr. Alice Thompson?"
        }}

        You should NOT incorporate any prior knowledge and should base the rewritten query only on the context and feedback provided.
        The `rewritten_input` MUST be a STRING.
        **

        Context:
        {context}

        Query:
        {original_query}

        Feedback:
        {feedback}

        JSON:
        """

    @staticmethod
    def generate_synthetic_scenarios(
        context: str,
        max_goldens_per_context: int,
        scenario_context: Optional[str],
        conversational_task: Optional[str],
        participant_roles: Optional[str],
    ):
        participant_section = (
            f"Each scenario MUST involve these participant roles: {participant_roles}."
            if participant_roles
            else "Each scenario MUST clearly specify who the participants are (e.g., 'a teacher and a student', 'two colleagues')."
        )

        scenario_context_section = (
            f"All scenarios MUST fit within this conversational context: {scenario_context}"
            if scenario_context
            else ""
        )

        task_section = (
            f"The conversation in each scenario should work towards this goal: {conversational_task}"
            if conversational_task
            else ""
        )

        return f"""I want you to act as a conversation scenario designer. Based on the given context, generate a list of JSON objects with a `scenario` key.
        Each `scenario` should describe a MULTI-TURN CONVERSATIONAL INTERACTION between specific participants discussing information from the context.

        **
        IMPORTANT: Please make sure to only return in JSON format, with the 'data' key as a list of JSON objects.
        You MUST TRY to generate {max_goldens_per_context} data points, unless scenarios become repetitive.

        Example context: ["Einstein won the Nobel Prize for his discovery of the photoelectric effect.", "Einstein won the Nobel Prize in 1921."]
        Example max goldens per context: 2
        Example JSON:
        {{
            "data": [
                {{
                    "scenario": "A high school student asks their physics teacher when Einstein won the Nobel Prize and what discovery it was awarded for"
                }},
                {{
                    "scenario": "Two university students are studying together and one tests the other's knowledge about Einstein's Nobel Prize year and the scientific work that earned it"
                }}
            ]  
        }}

        CRITICAL REQUIREMENTS FOR CONVERSATIONAL SCENARIOS:
        - Each scenario MUST describe a conversation between specific participants (who is talking to whom)
        - Each scenario MUST specify the conversational setting and context (where, why, what they're discussing)
        - DO NOT write questions, prompts, or instructions - write descriptions of conversational SITUATIONS
        - DO NOT use command phrases like "Explain...", "Compare...", "Describe..." - these are instructions, not conversations
        - Scenarios should describe realistic multi-turn interactions where information from context would naturally be discussed
        - Think: "Who is talking to whom, about what, and in what situation?"
        
        GOOD examples:
        ✓ "A patient asks their doctor about the side effects of a new medication during a consultation"
        ✓ "A manager provides feedback to an employee about their recent project performance in a 1-on-1 meeting"
        ✓ "Two friends debate the pros and cons of electric vehicles while carpooling to work"
        
        BAD examples (these are prompts/questions, not conversational scenarios):
        ✗ "Explain the side effects of medication"
        ✗ "What happens when water freezes?"
        ✗ "Compare electric vehicles to gas vehicles"
        
        You should NOT incorporate any prior knowledge you have and take each context at face value.
        {participant_section}
        {scenario_context_section}
        {task_section}
        You MUST TRY to generate {max_goldens_per_context} data points, unless scenarios become repetitive.
        **

        Max Goldens Per Context:
        {max_goldens_per_context}

        Context:
        {context}

        JSON:
        """

    @staticmethod
    def generate_synthetic_expected_outcome_conversational(
        scenario: str, context: str, expected_outcome_format: Optional[str]
    ):
        format_section = (
            f"The expected outcome MUST adhere to this format: {expected_outcome_format}"
            if expected_outcome_format
            else "Keep the expected outcome CONCISE (1-3 sentences maximum)"
        )

        return f"""Given the conversational scenario, generate a CONCISE expected outcome describing what should happen in the conversation or what is achieved by the end.

        **
        IMPORTANT: {format_section}
        
        The expected outcome should briefly describe ONE of:
        - What key information is shared/conveyed during the conversation
        - What the participants learn or come to understand
        - What decision, agreement, or resolution is reached
        - How the conversational goal is accomplished
        
        DO NOT write long explanatory paragraphs. Be direct and concise.
        Use information from the context to ground the expected outcome.
        **

        Context:
        {context}

        Conversational Scenario:
        {scenario}

        Expected Outcome:
        """

    @staticmethod
    def rewrite_evolved_scenario(
        evolved_scenario: str,
        scenario_context: Optional[str] = None,
        conversational_task: Optional[str] = None,
        participant_roles: Optional[str] = None,
    ):
        context_section = (
            f'Scenario Context: "{scenario_context}"'
            if scenario_context
            else ""
        )
        task_section = (
            f'Conversational Task: "{conversational_task}"'
            if conversational_task
            else ""
        )
        roles_section = (
            f'Participant Roles: "{participant_roles}"'
            if participant_roles
            else ""
        )

        return f"""Given the evolved scenario, which describes a conversational situation, generate a JSON object with a key 'scenario'. 
        This key should contain a scenario description that fits the provided context, aligns with the conversational task, and involves the specified participant roles (if provided).

        **
        IMPORTANT: Try to change the evolved scenario as little as possible. However, if it does not align with the provided scenario context, conversational task, or participant roles, it must be adjusted to fit these requirements. 
        
        The output must be in JSON format with the 'scenario' key only.
        The scenario MUST describe a conversational interaction, not a question or prompt.
        **

        Example Evolved Scenario: "Discuss the importance of meeting deadlines"
        Example Scenario Context: "Workplace performance management"
        Example Conversational Task: "Provide constructive feedback"
        Example Participant Roles: "Manager and employee"
        Example JSON: {{
            "scenario": "A manager meets with an employee to discuss recent missed deadlines and collaboratively develop strategies for better time management"
        }}

        Evolved Scenario:
        {evolved_scenario}
        
        {context_section}
        {task_section}
        {roles_section}

        JSON:
        """

    @staticmethod
    def rewrite_synthetic_scenarios(context, original_scenario, feedback):
        return f"""I want you to act as a scenario rewriter. Based on the provided context, original scenario, and feedback, generate a rewritten scenario that improves its clarity and conversational nature based on the feedback provided.

        **
        IMPORTANT: Please make sure to only return in JSON format, with the 'rewritten_scenario' key.
        The rewritten scenario MUST describe a conversational interaction between participants, not a question or instruction.

        Example context: "The Golden Gate Bridge, located in San Francisco, was completed in 1937 and is known for its Art Deco design. It connects the city of San Francisco to Marin County and spans the Golden Gate Strait."
        Example scenario: "Someone asks about a bridge"
        Example feedback: "The scenario is too vague and doesn't describe a conversational situation with specific participants. It should clearly identify who is talking to whom and in what context."
        Example JSON:
        {{
            "rewritten_scenario": "A tourist visiting San Francisco asks their tour guide about the history and design features of the Golden Gate Bridge"
        }}

        Example context: "The paper 'Advancements in Quantum Computing' by Dr. Alice Thompson discusses breakthroughs in quantum algorithms and was published in 2022. It explores the potential applications of quantum computing in cryptography and drug discovery."
        Example scenario: "A discussion about quantum computing"
        Example feedback: "The scenario lacks specificity about who is having the discussion, what the setting is, and what aspect they're focused on. Frame this as a concrete conversational situation with identified participants."
        Example JSON:
        {{
            "rewritten_scenario": "A graduate student presents Dr. Alice Thompson's 2022 paper on quantum computing to their research group, leading to a discussion about applications in cryptography and drug discovery"
        }}

        You should NOT incorporate any prior knowledge and should base the rewritten scenario only on the context and feedback provided.
        The `rewritten_scenario` MUST be a STRING describing a multi-turn conversational interaction between specific participants.
        **

        Context:
        {context}

        Original Scenario:
        {original_scenario}

        Feedback:
        {feedback}

        JSON:
        """


######################################################################################################
##### Filter #########################################################################################
######################################################################################################


class FilterTemplate:

    @staticmethod
    def evaluate_synthetic_inputs(query):
        return f"""Evaluate the provided synthetic query (which may be a question, task, or instruction) for clarity and answerability, assuming sufficient domain knowledge. Use the following criteria to guide your assessment:

        1. **Self-Containment**: Can the query be understood and completed without needing additional context or external references not provided within the query itself? It should be self-sufficient, meaning it doesn't depend on specific documents, tables, or prior knowledge not included in the query.
        2. **Clear Objective**: Does the query clearly convey its intent? It should specify what information, action, or response is being requested, allowing for a direct and appropriate answer or execution without ambiguity.

        Based on these criteria, assign a score between 0 and 1, where:
        - "1" means the query is clear, self-contained, and answerable.
        - "0" means the query is vague, relies on external references, or is unclear in its intent.
        - Scores between 0 and 1 indicate partial clarity or answerability, where the query meets some but not all of the criteria.

        **
        IMPORTANT: Please make sure to only return in JSON format, with the 'feedback' and 'score' keys.

        Example query: "What technological innovations have changed communication over the last 20 years?"
        Example JSON:
        {{
            "feedback": "The query is somewhat vague as it asks about 'technological innovations' without specifying particular areas of communication (e.g., social media, messaging apps). It could be improved by narrowing the focus to a specific type of innovation or timeframe.",
            "score": 0.5
        }}

        Example query: "Explain the impact of renewable energy policies in Germany on local economies in 2021."
        Example JSON:
        {{
            "feedback": "This query clearly specifies the focus (renewable energy policies), the region (Germany), and the timeframe (2021). It is self-contained and answerable without needing additional context, making it clear and effective.",
            "score": 1.0
        }}

        Example query: "What are the main criticisms of the current education system in the United States?"
        Example JSON:
        {{
            "feedback": "The question is broad and lacks specificity, as 'main criticisms' could refer to various aspects (e.g., funding, curriculum, access). To improve clarity, it could specify which aspect of the education system is being critiqued.",
            "score": 0.4
        }}

        Example query: "Discuss the role of AI in healthcare, particularly in diagnostics, as noted in the last report."
        Example JSON:
        {{
            "feedback": "This question refers to 'the last report' without providing context or details, making it unclear and dependent on external information. It would be clearer if it provided some background on the report or defined what aspects of AI in diagnostics to address.",
            "score": 0.3
        }}
                
        The `feedback` MUST be a STRING and `score` must be a float from 0 to 1.
        **
                
        Query:
        {query}

        JSON:
        """

    @staticmethod
    def evaluate_context(context):
        return f"""Given a context, complete the following task and return the result in VALID JSON format: Evaluate the supplied context and assign a numerical score between 0 (Low) and 1 (High) for each of the following criteria in your JSON response:

        - **clarity**: Assess how clear and comprehensible the information is. A score of 1 indicates that the context is straightforward and easily understandable, while a score of 0 reflects vagueness or confusion in the information presented.
        - **depth**: Evaluate the extent of detailed analysis and the presence of original insights within the context. A high score (1) suggests a thorough and thought-provoking examination, while a low score (0) indicates a shallow overview of the subject.
        - **structure**: Review how well the content is organized and whether it follows a logical progression. A score of 1 is given to contexts that are coherently structured and flow well, whereas a score of 0 is for those that lack organization or clarity in their progression.
        - **relevance**: Analyze the importance of the content in relation to the main topic, awarding a score of 1 for contexts that stay focused on the subject without unnecessary diversions, and a score of 0 for those that include unrelated or irrelevant information.

        **
        IMPORTANT: Please make sure to only return in JSON format, with the 'clarity', 'depth', 'structure', and 'relevance' keys.

        Example context: "Artificial intelligence is rapidly changing various sectors, from healthcare to finance, by enhancing efficiency and enabling better decision-making."
        Example JSON:
        {{
            "clarity": 1,
            "depth": 0.8,
            "structure": 0.9,
            "relevance": 1
        }}

        Example context: "Cats are great pets. They like to sleep and play."
        Example JSON:
        {{
            "clarity": 0.5,
            "depth": 0.3,
            "structure": 0.4,
            "relevance": 0.5
        }}

        Example context: "Artificial intelligence is rapidly changing various sectors, from healthcare to finance, by enhancing efficiency and enabling better decision-making."
        Example JSON:
        {{
            "clarity": 1,
            "depth": 0.9,
            "structure": 1,
            "relevance": 1
        }}

        Example context: "Artificial intelligence is rapidly changing various sectors, from healthcare to finance, by enhancing efficiency and enabling better decision-making."
        Example JSON:
        {{
            "clarity": 0.4,
            "depth": 0,
            "structure": 0.3,
            "relevance": 0.2
        }}

        Example context: "The impact of globalization on local cultures is complex, with both positive and negative effects. It can lead to cultural exchange but also to the erosion of local traditions."
        Example JSON:
        {{
            "clarity": 0.9,
            "depth": 0.8,
            "structure": 0.9,
            "relevance": 1
        }}


        `clarity`, `depth`, `structure`, and `relevance` MUST be floats from 0 to 1.
        Make sure your JSON response is valid and properly formatted.
        **

        context:
        {context}

        JSON:
        """

    @staticmethod
    def evaluate_synthetic_scenarios(scenario):
        return f"""Evaluate the provided conversational scenario for clarity, conversational nature, and appropriateness. Use the following criteria:

        1. **Conversational Structure**: Does the scenario describe an actual conversation between identified participants (not just a question or prompt)?
        2. **Participant Clarity**: Are the participants clearly identified with specific roles (e.g., "teacher and student", "doctor and patient")?
        3. **Contextual Setting**: Is there a clear setting or context for when/where/why this conversation occurs?
        4. **Purposeful Interaction**: Does the scenario imply a multi-turn dialogue with a goal or purpose?
        5. **Naturalness**: Could this conversation realistically occur in the described situation?

        Assign a score between 0 and 1:
        - "1" = Perfect conversational scenario with clear participants, setting, and purpose
        - "0.7-0.9" = Good scenario with minor issues (slightly vague participants or context)
        - "0.4-0.6" = Mediocre scenario (missing clear participants OR setting OR purpose)
        - "0-0.3" = Poor scenario (just a question/prompt, or very vague)

        **
        IMPORTANT: Return JSON format only, with 'feedback' and 'score' keys.

        Example scenario: "A student asks about homework"
        Example JSON:
        {{
            "feedback": "This scenario is too vague. It doesn't specify what subject, what specific aspect of homework, who the student is asking (teacher? parent?), or provide conversational context. It reads more like a prompt fragment than a conversational scenario. Needs: specific participants, setting, and what aspect of homework is being discussed.",
            "score": 0.2
        }}

        Example scenario: "A new employee meets with their supervisor for a quarterly performance review to discuss progress over the first three months, areas for improvement, and goals for the next quarter"
        Example JSON:
        {{
            "feedback": "Excellent conversational scenario. Clear participants (new employee and supervisor), specific setting (quarterly performance review), clear purpose (discuss progress, improvements, and goals), and describes a realistic multi-turn conversation. This would naturally involve back-and-forth dialogue.",
            "score": 1.0
        }}

        Example scenario: "A patient explains symptoms to a doctor during a check-up"
        Example JSON:
        {{
            "feedback": "Good conversational scenario with clear participants (patient and doctor) and setting (check-up). It describes a realistic interaction. Could be slightly more specific about what symptoms or the purpose beyond just explaining, but overall solid.",
            "score": 0.8
        }}

        Example scenario: "Explain how photosynthesis works"
        Example JSON:
        {{
            "feedback": "This is an instruction/prompt, not a conversational scenario. It doesn't describe who is talking to whom, what the setting is, or frame it as an interaction. Needs complete rewrite to describe an actual conversation (e.g., 'A biology teacher explains photosynthesis to a student who asks about...').",
            "score": 0.1
        }}

        Example scenario: "Two colleagues discuss the benefits and drawbacks of remote work versus office work during their lunch break"
        Example JSON:
        {{
            "feedback": "Strong conversational scenario. Clear participants (two colleagues), setting (lunch break), specific topic (remote vs office work), and implies a natural back-and-forth discussion. Realistic and purposeful.",
            "score": 0.95
        }}
                
        The `feedback` MUST be a STRING and `score` must be a float from 0 to 1.
        **
                
        Scenario:
        {scenario}

        JSON:
        """


######################################################################################################
##### Approach similar to https://github.com/nlpxucan/WizardLM/blob/main/Evol_Instruct/depth.py ######
######################################################################################################


class EvolutionTemplate:

    base_instruction = """I want you to act as an input rewriter.
    Your object is the rewrite a given `input` and must be factually correct according to the supporting information in `Context`.
    You MUST complicate the given `Input` using the following method:"""

    @staticmethod
    def multi_context_evolution(input, context):
        return (
            EvolutionTemplate.base_instruction
            + f"""
            1. `Input` should be rewritten to require readers to use information from all elements of `Context`. 
            2. `Rewritten Input` must be fully answerable from information in `Context`. 
            3. `Rewritten Input` should be concise and understandable by humans.
            4. `Rewritten Input` should not contain phrases like  'based on the provided context' or 'according to the context'.
            5. `Rewritten Input` should not contain more than 15 words. Use abbreviation wherever possible.
            
            **
            EXAMPLES

            Example context:
            ["Vaccines introduce a weakened or dead form of the pathogen to the human body.", "This exposure helps the immune system learn to recognize and fight the pathogen in the future."]
            Example input:
            How do vaccines work?
            Example rewritten input:
            How does the introduction of a modified pathogen prepare the immune system for future encounters?

            --------------------------
            
            Example context:
            ["Plants perform photosynthesis, using sunlight to convert carbon dioxide and water into glucose and oxygen.", "Chlorophyll in plant leaves absorbs sunlight, initiating the photosynthesis process.", "Oxygen is a by-product of the photosynthesis process and is released into the atmosphere."]
            Example input:
            Explain how plants produce oxygen.
            Example rewritten input: 
            Considering chlorophyll's role in sunlight absorption and photosynthesis, how is oxygen produced and released by plants?

            --------------------------

            Example context:
            ["The gravitational pull of the moon on the Earth influences the tides.", "The position of the sun relative to the Earth and the moon also affects tidal patterns."]
            Example input:
            Tell me about high tides.
            Example rewritten input:
            Explain how the combined gravitational effects of the moon and the sun's relative positioning influence Earth's tidal phenomena.
            **

            Context:
            {context}
            Input:
            {input}
            Rewritten Input:            
            """
        )

    @staticmethod
    def reasoning_evolution(input, context):
        return (
            EvolutionTemplate.base_instruction
            + f"""
            1. If `Input` can be solved with just a few simple thinking processes, you can rewrite it to explicitly request multiple-step reasoning.
            2. `Rewritten Input` should require readers to make multiple logical connections or inferences.
            3. `Rewritten Input` should be concise and understandable by humans.
            4. `Rewritten Input` should not contain phrases like  'based on the provided context' or 'according to the context'.
            5. `Rewritten Input` must be fully answerable from information in `Context`. 
            6. `Rewritten Input` should not contain more than 15 words. Use abbreviation wherever possible.

            **
            EXAMPLES

            Example context:
            Chlorophyll allows plants to absorb energy from light, and this energy is used to convert carbon dioxide and water into glucose and oxygen, a process known as photosynthesis.
            Example input:
            Why are plants green?
            Example rewritten input:
            How does chlorophyll's role in absorbing light relate to plants' green color and their ability to produce glucose?
        
            --------------------------
            
            Example context:
            The greenhouse effect occurs when the Earth's atmosphere traps solar radiation, caused by gases such as carbon dioxide, methane, and water vapor. This process maintains the planet's temperature but can lead to increased global temperatures when exacerbated by human activities.
            Example input:
            What causes seasons to change?
            Example rewritten input: 
            Given the trapping of solar radiation by atmospheric gases, explain how the enhanced activity impacts Earth's climate.

            --------------------------

            Example context:
            Economic theories suggest that market demand and supply determine prices, but government policies can also influence market dynamics through regulations, taxes, and subsidies.
            Example input:
            Identify the primary factors that determine the price of goods in a market.
            Example rewritten input:
            Examine how the interplay of market demand, supply dynamics, and government policy interventions collectively shapes the pricing mechanism of goods within a market ecosystem.
            **

            Context:
            {context}
            Input:
            {input}
            Rewritten Input:            
            """
        )

    @staticmethod
    def concretizing_evolution(input, context):
        return (
            EvolutionTemplate.base_instruction
            + f"""
            1. Rewrite `Input` by replacing general concepts/inquiries with more specific ones.
            2. `Rewritten Input` should be concise and understandable by humans.
            3. `Rewritten Input` should not contain phrases like  'based on the provided context' or 'according to the context'.
            4. `Rewritten Input` must be fully answerable from information in `Context`.  
            5. `Rewritten Input` should not contain more than 15 words. Use abbreviation wherever possible.

            **
            EXAMPLES
            Example context:
            Rainforests are home to over half of the world's plant and animal species, making them key to maintaining global biodiversity. The variety of life found in these ecosystems contributes to genetic diversity, which is crucial for adaptation and survival amid changing environmental conditions. This biodiversity also supports ecosystem resilience, enabling forests to recover from disturbances.
            The biodiversity in rainforests plays a significant role in human well-being, providing essential services such as air and water purification, disease control, and pollination of crops. Additionally, many medicines are derived from rainforest plants, highlighting the importance of these ecosystems for medical research and healthcare.
            Example input: 
            Why is the biodiversity of rainforests important?
            Example rewritten input:
            How does the extensive biodiversity found in rainforests, encompassing over half of the world's plant and animal species, contribute to global biodiversity maintenance, and what role does this diversity play in enhancing ecosystem resilience, human health through disease control, crop pollination, and the development of medicines derived from rainforest plants?

            --------------------------

            Example context:
            Bees play a critical role in pollinating flowering plants, including many fruits and vegetables, contributing to the diversity of plant life and the production of crops. Their activity supports the growth of trees, flowers, and other plants, which serve as food and shelter for numerous animals, thus maintaining ecosystem balance.
            Beyond their impact on food crops, bees contribute to wild plant growth by pollinating a wide range of plants outside of agricultural settings. This pollination is vital for the reproduction of many plants, affecting entire ecosystems' health and sustainability.
            Example input: 
            What is the role of bees in ecosystems?
            Example rewritten input:
            How do bees, through their pollination of flowering plants, including a multitude of fruits and vegetables, significantly influence the diversity of plant life and agricultural productivity, and in what ways do their activities extend beyond agricultural settings to support the growth of trees, flowers, and other plants, thereby providing essential resources for various animal species and contributing to the overall balance and sustainability of ecosystems?

            --------------------------

            Example context:
            Solar power generation relies on photovoltaic cells to convert sunlight into electricity. These cells are made of materials that exhibit the photovoltaic effect, which occurs when light photons are absorbed by the material, causing the generation of electrical current.
            Solar panels, composed of many photovoltaic cells, collect sunlight and convert it into electrical power. This energy can then be used directly or stored in batteries for later use, providing a renewable and sustainable source of power with minimal environmental impact.
            Example input: 
            What are the principles behind solar power generation?
            Example rewritten input:
            How do photovoltaic cells work to convert sunlight into electrical power, and what role do solar panels play in this process, including energy storage for sustainable use?
            **

            Input:
            {input}
            Context:
            {context}
            Rewritten Input:
            """
        )

    @staticmethod
    def constrained_evolution(input, context):
        return (
            EvolutionTemplate.base_instruction
            + f"""
            1. Rewrite `Input` by adding at least one more constraints/requirements.
            2. `Rewritten Input` must be fully answerable from information in `Context`. 
            5. `Rewritten Input` should not contain more than 15 words. Use abbreviation wherever possible.

            **
            EXAMPLES
            Example context:
            Rainforests are home to over half of the world's plant and animal species, making them key to maintaining global biodiversity. The variety of life found in these ecosystems contributes to genetic diversity, which is crucial for adaptation and survival amid changing environmental conditions. This biodiversity also supports ecosystem resilience, enabling forests to recover from disturbances.
            The biodiversity in rainforests plays a significant role in human well-being, providing essential services such as air and water purification, disease control, and pollination of crops. Additionally, many medicines are derived from rainforest plants, highlighting the importance of these ecosystems for medical research and healthcare.
            Example input: 
            Why is the biodiversity of rainforests important?
            Example rewritten input:
            How does the biodiversity of rainforests contribute to ecosystem resilience and recovery from disturbances, and in what ways does it impact human well-being through services such as air and water purification, disease control, and crop pollination?

            --------------------------

            Example context:
            Bees play a critical role in pollinating flowering plants, including many fruits and vegetables, contributing to the diversity of plant life and the production of crops. Their activity supports the growth of trees, flowers, and other plants, which serve as food and shelter for numerous animals, thus maintaining ecosystem balance.
            Beyond their impact on food crops, bees contribute to wild plant growth by pollinating a wide range of plants outside of agricultural settings. This pollination is vital for the reproduction of many plants, affecting entire ecosystems' health and sustainability.
            Example input: 
            What is the role of bees in ecosystems?
            Example rewritten input:
            Considering the pivotal role bees play in pollinating both agricultural crops and wild plants, thereby contributing to the diversity of plant life and supporting the foundation of food chains, analyze how bees influence the growth and sustainability of various ecosystems.

            --------------------------

            Example context:
            Solar power generation relies on photovoltaic cells to convert sunlight into electricity. These cells are made of materials that exhibit the photovoltaic effect, which occurs when light photons are absorbed by the material, causing the generation of electrical current.
            Solar panels, composed of many photovoltaic cells, collect sunlight and convert it into electrical power. This energy can then be used directly or stored in batteries for later use, providing a renewable and sustainable source of power with minimal environmental impact.
            Example input: 
            What are the principles behind solar power generation?
            Example rewritten input:
            Explain how photovoltaic cells convert sunlight into electricity and discuss the environmental benefits of solar power as a sustainable energy source.
            **

            Context:
            {context}
            Input:
            {input}
            Rewritten Input:
            """
        )

    @staticmethod
    def comparative_question_evolution(input, context):
        return (
            EvolutionTemplate.base_instruction
            + f"""
            1. Rewrite `Input` to focus on comparing two or more entities, concepts, or processes.
            2. `Rewritten Input` should encourage a detailed comparison that highlights similarities and differences.
            3. `Rewritten Input` must be fully answerable from information in `Context`. 
            4. `Rewritten Input` should be concise and understandable by humans.
            5. `Rewritten Input` should not contain phrases like  'based on the provided context' or 'according to the context'.
            6. `Rewritten Input` should not contain more than 15 words. Use abbreviation wherever possible.

            **
            EXAMPLES
            Example context:
            "Water boils at 100°C (212°F) at sea level, but boiling point decreases with altitude due to lower atmospheric pressure. In contrast, alcohol boils at about 78°C (172°F)."
            Example input: 
            What happens to water as it boils?
            Example rewritten input:
            How does the boiling point of water at sea level compare to that of alcohol, and how does altitude affect water's boiling point?

            --------------------------

            Example context:
            "Photosynthesis in plants involves converting carbon dioxide and water into glucose and oxygen, using sunlight. Cellular respiration in animals converts glucose and oxygen back into carbon dioxide and water, releasing energy."
            Example input: 
            How do plants and animals process energy?
            Example rewritten input:
            Compare the processes of photosynthesis in plants and cellular respiration in animals, focusing on inputs and outputs of each process.

            --------------------------

            Example context:
            "The Renaissance was a period of significant cultural, artistic, and scientific rebirth that began in the 14th century, primarily in Italy. The Enlightenment, occurring mainly in the 18th century, centered around reason, science, and individualism, significantly influencing European thought."
            Example input: 
            What was the Renaissance?
            Example rewritten input:
            Contrast the main focuses and impacts of the Renaissance and the Enlightenment on European thought and culture.

            --------------------------

            Context:
            {context}
            Input:
            {input}
            Rewritten Input:
            """
        )

    @staticmethod
    def hypothetical_scenario_evolution(input, context):
        return (
            EvolutionTemplate.base_instruction
            + f"""
            1. Rewrite `Input` to include a hypothetical or speculative scenario that is relevant to the `Context`.
            2. `Rewritten Input` should encourage the reader to apply knowledge from the `Context` to imagine or deduce outcomes.
            3. `Rewritten Input` should be concise, clear, and understandable by humans.
            4. `Rewritten Input` should not contain phrases like 'based on the provided context' or 'according to the context'.
            5. `Rewritten Input` must be fully answerable from information in `Context`.
            6. `Rewritten Input` should not contain more than 15 words. Use abbreviation wherever possible.

            **
            EXAMPLES

            Example context:
            The greenhouse effect is a natural process where the Earth's atmosphere traps some of the Sun's energy, warming the planet to a temperature that supports life. Human activities, particularly the emission of greenhouse gases like carbon dioxide and methane, have intensified this effect, leading to global warming and climate change.
            Example input:
            What are the consequences of the greenhouse effect?
            Example rewritten input:
            Imagine a world where greenhouse gas emissions were doubled overnight. How might this intensified greenhouse effect impact global climate patterns and ecosystems?

            --------------------------

            Example context:
            Antibiotics are drugs used to treat bacterial infections. They work by killing bacteria or preventing their growth. However, overuse and misuse of antibiotics have led to the development of antibiotic-resistant bacteria, which are harder to treat because they can withstand the drugs designed to kill them.
            Example input:
            How do antibiotics work?
            Example rewritten input:
            In a scenario where a new antibiotic-resistant superbug emerges, how would the principles of antibiotic action and resistance influence our approach to treatment?

            --------------------------

            Example context:
            Quantum computing relies on the principles of quantum mechanics to process information, utilizing quantum bits or qubits. These qubits can exist in multiple states simultaneously, allowing quantum computers to perform complex calculations much faster than traditional computers.
            Example input:
            What is quantum computing?
            Example rewritten input:
            Suppose a quantum computer was tasked with solving a problem that currently takes traditional computers centuries to solve. How might the unique capabilities of quantum computing change the outcome?
            **

            Context:
            {context}
            Input:
            {input}
            Rewritten Input:
            """
        )

    @staticmethod
    def in_breadth_evolution(input, context):
        return (
            EvolutionTemplate.base_instruction
            + f"""
            1. Rewrite `Input` to create a brand new prompt.
            2. `Rewritten Input` should belong to the same domain as the `input` but be even more rare.
            3. `Rewritten Input` should be concise, clear, and understandable by humans.
            4. `Rewritten Input` should not contain phrases like 'based on the provided context' or 'according to the context'.
            5. `Rewritten Input` should not contain more than 15 words. Use abbreviation wherever possible.

            **
            EXAMPLES

            Example context:
            Wearable technology has revolutionized personal health monitoring, allowing individuals to track vital signs and activity levels in real time.
            Example input:
            Explore the impact of wearable technology on personal health management.
            Example rewritten input:
            Delve into the development of implantable health devices and their potential to transform chronic disease management.

            --------------------------

            Example context:
            Quantum computing leverages the principles of quantum mechanics to process information, offering significant advancements over traditional computing methods.
            Example input:
            How is quantum computing different from traditional computing?
            Example rewritten input:
            Explore the potential of quantum cryptography in enhancing cybersecurity measures beyond current encryption standards

            --------------------------

            Example context:
            Virtual reality (VR) offers immersive learning experiences, transforming educational methodologies by providing interactive and engaging ways to acquire knowledge, especially in fields requiring practical skills.
            Example input:
            What impact does virtual reality (VR) have on education?
            Example rewritten input:
            Investigate the use of VR simulations in medical training to enhance practical skills and decision-making under pressure.
            **

            Context:
            {context}
            Input:
            {input}
            Rewritten Input:
            """
        )


class ConversationalEvolutionTemplate:

    base_instruction = """I want you to act as a conversational scenario rewriter.
    Your objective is to rewrite the given `Scenario` while preserving factual correctness according to the supporting information in `Context`.
    You MUST complicate the given `Scenario` using the following method:"""

    @staticmethod
    def multi_context_evolution(scenario, context):
        return (
            ConversationalEvolutionTemplate.base_instruction
            + f"""
            1. `Scenario` must be rewritten so participants must naturally rely on **all elements of `Context`** during the conversation.
            2. `Rewritten Scenario` MUST remain a realistic multi-turn conversation setup.
            3. Keep the rewritten scenario under **60 words**.
            4. Do NOT use phrases like “based on the context” or “according to the context”.

            **
            EXAMPLES

            Example context:
            ["A startup is developing an AI tool for diagnosing skin conditions.",
             "Regulations require explainability for clinical AI systems.",
             "The team is under a tight deadline before a regulatory audit."]
            Example scenario:
            Two engineers review their prototype.
            Example rewritten scenario:
            During a tense late-night meeting, two AI engineers debate whether their skin-diagnosis model meets upcoming explainability regulations, forcing them to discuss audit risks and integrate overlooked clinical requirements across multiple conversational turns.

            --------------------------

            Example context:
            ["A research team is studying coral bleaching.",
             "Rising ocean temperatures accelerate bleaching events.",
             "Funding depends on publishing actionable mitigation strategies."]
            Example scenario:
            Two scientists talk about coral reefs.
            Example rewritten scenario:
            In a lab debrief, two marine biologists argue over how rising ocean temperatures, bleaching data, and funding-dependent mitigation strategies should shape their next field report.

            **

            Context:
            {context}
            Scenario:
            {scenario}
            Rewritten Scenario:
            """
        )

    @staticmethod
    def reasoning_evolution(scenario, context):
        return (
            ConversationalEvolutionTemplate.base_instruction
            + f"""
            1. Rewrite `Scenario` so the resulting conversation requires multi-step reasoning between participants.
            2. Add layered inferential or analytical demands grounded in `Context`.
            3. Keep the rewritten scenario under **60 words**.
            4. Do NOT use phrases like “based on the context”.
            5. Must remain a realistic multi-turn dialogue setup.

            **
            EXAMPLES

            Example context:
            "A school is transitioning to solar power, but initial costs are high and maintenance requires specialized knowledge."
            Example scenario:
            A teacher asks a technician about solar panels.
            Example rewritten scenario:
            A teacher and campus technician debate whether adopting solar panels makes financial sense, analyzing upfront costs, long-term energy savings, and specialized maintenance requirements across a multi-step reasoning exchange.

            --------------------------

            Example context:
            "An economic model predicts inflation rises when supply chains weaken."
            Example scenario:
            Two analysts discuss inflation.
            Example rewritten scenario:
            On a strategy call, two analysts unpack how supply-chain disruptions, demand shifts, and model predictions interact, forcing a layered reasoning conversation.

            **

            Context:
            {context}
            Scenario:
            {scenario}
            Rewritten Scenario:
            """
        )

    @staticmethod
    def concretizing_evolution(scenario, context):
        return (
            ConversationalEvolutionTemplate.base_instruction
            + f"""
            1. Rewrite `Scenario` by replacing general conversational settings with **highly specific**, concrete circumstances tied to `Context`.
            2. Add situational cues, named events, or explicit constraints.
            3. Keep the rewritten scenario under **60 words**.
            4. Maintain realistic multi-turn dialogue structure.

            **
            EXAMPLES

            Example context:
            "A hospital is piloting a new triage AI system."
            Example scenario:
            A doctor and nurse discuss patient triage.
            Example rewritten scenario:
            During a chaotic evening shift, a doctor and nurse debate whether the new triage AI's risk-scores should override manual judgment in handling a surge of incoming trauma cases.

            --------------------------

            Example context:
            "A remote-work company is struggling with meeting overload."
            Example scenario:
            Two colleagues discuss productivity.
            Example rewritten scenario:
            In a Friday retrospective, two remote employees argue about whether asynchronous updates can replace their current schedule of back-to-back video meetings.

            **

            Context:
            {context}
            Scenario:
            {scenario}
            Rewritten Scenario:
            """
        )

    @staticmethod
    def constrained_evolution(scenario, context):
        return (
            ConversationalEvolutionTemplate.base_instruction
            + f"""
            1. Rewrite `Scenario` by adding at least **one new constraint** that shapes how the conversation unfolds.
            2. The constraint must logically follow from `Context`.
            3. Keep the rewritten scenario under **60 words**.
            4. Keep it a realistic multi-turn setup.

            **
            EXAMPLES

            Example context:
            "A startup must deliver an AI model but cannot exceed strict GPU budgets."
            Example scenario:
            Two engineers discuss model performance.
            Example rewritten scenario:
            Two ML engineers debate model redesigns, but GPU usage is capped for the quarter, forcing them to reconsider heavier architectures while under deadline pressure.

            --------------------------

            Example context:
            "A university's ethics board is reviewing data-collection policies."
            Example scenario:
            A professor talks with a student researcher.
            Example rewritten scenario:
            Before submitting their study, a professor and student must revise their protocol to satisfy strict new privacy constraints imposed by the ethics board.

            **

            Context:
            {context}
            Scenario:
            {scenario}
            Rewritten Scenario:
            """
        )

    @staticmethod
    def comparative_question_evolution(scenario, context):
        return (
            ConversationalEvolutionTemplate.base_instruction
            + f"""
            1. Rewrite `Scenario` so the conversation naturally compares two or more concepts, tools, or approaches.
            2. The comparison must be central to the multi-turn dialogue.
            3. Keep the rewritten scenario under **60 words**.

            **
            EXAMPLES

            Example context:
            "Two project management tools differ in cost, automation, and integration options."
            Example scenario:
            Two coworkers evaluate a new tool.
            Example rewritten scenario:
            In a planning meeting, two coworkers compare switching from their legacy management tool to a cheaper automated one, weighing integration gaps and workflow impact.

            --------------------------

            Example context:
            "Electric and hydrogen vehicles have different refueling logistics."
            Example scenario:
            Two friends discuss cars.
            Example rewritten scenario:
            On a road trip, two friends debate electric vs hydrogen cars, comparing range limits, refueling times, and long-term reliability.

            **

            Context:
            {context}
            Scenario:
            {scenario}
            Rewritten Scenario:
            """
        )

    @staticmethod
    def hypothetical_scenario_evolution(scenario, context):
        return (
            ConversationalEvolutionTemplate.base_instruction
            + f"""
            1. Rewrite `Scenario` by adding a hypothetical twist grounded in `Context`.
            2. The speculative change MUST drive the conversation.
            3. Must remain realistic and multi-turn.
            4. Keep the rewritten scenario under **60 words**.

            **
            EXAMPLES

            Example context:
            "A cybersecurity team is tracking frequent phishing attempts."
            Example scenario:
            Two analysts review security logs.
            Example rewritten scenario:
            During a nightly shift, two analysts discuss a hypothetical spike in coordinated phishing attacks and explore how it would strain their current detection pipeline.

            --------------------------

            Example context:
            "A city is experimenting with autonomous buses."
            Example scenario:
            A resident talks to a planner.
            Example rewritten scenario:
            At a community forum, a resident and transit planner imagine a scenario where all local buses become autonomous overnight and debate safety tradeoffs.

            **

            Context:
            {context}
            Scenario:
            {scenario}
            Rewritten Scenario:
            """
        )

    @staticmethod
    def in_breadth_evolution(scenario, context):
        return (
            ConversationalEvolutionTemplate.base_instruction
            + f"""
            1. Rewrite `Scenario` into a brand-new conversational setup.
            2. It must remain in the **same domain** but shift toward a **rarer or niche** topic.
            3. Must remain a realistic multi-turn dialogue setup.
            4. Keep under **60 words**.

            **
            EXAMPLES

            Example context:
            "Wearables monitor heart rate and sleep cycles."
            Example scenario:
            Two people discuss fitness trackers.
            Example rewritten scenario:
            In a clinical trial briefing, two researchers debate implantable cardiac micro-sensors and their potential to outperform traditional wearables in long-term monitoring.

            --------------------------

            Example context:
            "Quantum computing is advancing rapidly."
            Example scenario:
            Two students study quantum algorithms.
            Example rewritten scenario:
            During a research seminar, two students examine the niche topic of quantum-secure error-correcting codes for next-generation cryptosystems.

            **

            Context:
            {context}
            Scenario:
            {scenario}
            Rewritten Scenario:
            """
        )
