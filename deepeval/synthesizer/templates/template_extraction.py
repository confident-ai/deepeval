class ExtractionTemplate:

    @staticmethod
    def extract_prompt_structure_from_inputs(inputs: list[str]):
        return f"""
            You are a prompt engineer tasked with reverse-engineering the original prompt that may have produced the following inputs. 
            Each input is a message that a user might submit to an AI system.

            Your job is to infer the structure of the original prompt by analyzing patterns in these inputs.

            Specifically, extract the following:
            
            1. `scenario`: Describe the type of person or user who would have submitted these inputs, and the context or purpose for doing so.
            2. `task`: What was the AI system expected to do in response to these inputs?
            3. `input_format`: Describe the style, tone, or structure of the inputs — how the inputs are typically phrased.

            You MUST return your answer strictly in the following JSON format:

            ```json
            {{
                "scenario": "<your answer here>",
                "task": "<your answer here>",
                "input_format": "<your answer here>"
            }}
            ```

            **
            IMPORTANT: Do not use any prior knowledge. Only rely on what is observable in the inputs themselves.   

            Example inputs: [
                "How many users signed up last week?",
                "Show me the total revenue for March.",
                "Which products had the highest sales yesterday?"
            ]

            Example output:
            {{
                "scenario": "Non-technical users trying to query a database using plain English.",
                "task": "Answering text-to-SQL-related queries by querying a database and returning the results to users.",
                "input_format": "Questions in English that ask for data in a database."
            }}

            Here are the inputs to analyze:

            {inputs}
     """

    @staticmethod
    def extract_conversational_structure_from_scenarios(example_scenarios):
        scenarios_text = "\n".join(
            [f"- {scenario}" for scenario in example_scenarios]
        )

        return f"""Analyze the following conversational scenarios and extract the common structural elements:

        Example Scenarios:
        {scenarios_text}

        Based on these examples, identify and return in JSON format:
        1. **scenario_context**: The general context or domain in which these conversations occur (e.g., "customer service", "educational settings", "workplace interactions")
        2. **conversational_task**: The primary goal or purpose these conversations aim to achieve (e.g., "resolve issues", "provide information", "give feedback")
        3. **participant_roles**: The typical participants involved in these conversations (e.g., "customer and support agent", "teacher and student", "manager and employee")

        **
        IMPORTANT: Please make sure to only return in JSON format, with the 'scenario_context', 'conversational_task', and 'participant_roles' keys.

        Example JSON:
        {{
            "scenario_context": "Educational settings and academic discussions",
            "conversational_task": "Explain concepts and answer questions",
            "participant_roles": "Teacher and student, or peer students"
        }}

        The values MUST be STRINGS that capture the essence of the conversational patterns in the examples.
        **

        JSON:
        """
