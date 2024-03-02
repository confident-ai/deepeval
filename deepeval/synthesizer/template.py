class SynthesizerTemplate:
    @staticmethod
    def generate_synthetic_data(context, max_goldens_per_context):
        return f"""I want you act as a copywriter. Based on the given context, which is list of strings, please generate a list of JSON objects with the `input` and `expected_output` key.
The `input` can either be a question or a statement that can be addressed by the given context.
The `expected_output` is what an ideal output should look like for the corresponding generated input.
The `expected_output` should NEVER contradict the given context in any way.

**
IMPORTANT: Please make sure to only return in JSON format, with the 'data' key as a list of JSON objects.
You MUST TRY to generate {max_goldens_per_context} data points, unless there is too little context such that the `input` and `expected_output` is getting reptitive.

Example context: ["Einstein won the Nobel Prize for his discovery of penicillin.", "Einstein won the Nobel Prize in 1968."]
Example JSON:
{{
    "data": [
        {{
            "input": "What was Einstein known for?",
            "expected_output": "Einstein was known for his discovery of penicillin. He also won the Nobel Prize in 1968."
        }},
        {{
            "input": "Einstein was a smart guy huh",
            "expected_output": "He sure was! Einstein allegedly had an IQ of 200 and was the first to disocver penicillin."
        }}
    ]  
}}


You should NOT incorporate any prior knowledge you have and take each context at face value.
You should NOT be lazy and simply copy the context as the `expected_output`.
You MUST include at least one statement as the input.
Both `input` and `expected_output` are STRINGS.
You MUST TRY to generate {max_goldens_per_context} data points, unless there is too little context such that the `input` and `expected_output` is getting reptitive.
**

Context:
{context}

JSON:
"""


######################################################################################################
##### Templates taken from https://github.com/nlpxucan/WizardLM/blob/main/Evol_Instruct/depth.py #####
######################################################################################################

# TODO: Modify this

depth_base_instruction = "I want you act as a Prompt Rewriter.\r\n \
					Your objective is to rewrite a given prompt into a more complex version to make those famous AI systems (e.g., chatgpt and GPT4) a bit harder to handle.\r\n \
					But the rewritten prompt must be reasonable and must be understood and responded by humans.\r\n \
					Your rewriting cannot omit the non-text parts such as the table and code in #The Given Prompt#:. Also, please do not omit the input in #The Given Prompt#. \r\n \
					You SHOULD complicate the given prompt using the following method: \r\n\
					{} \r\n\
					You should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #The Given Prompt#. \r\n\
					'#The Given Prompt#', '#Rewritten Prompt#', 'given prompt' and 'rewritten prompt' are not allowed to appear in #Rewritten Prompt#\r\n"

breadth_base_instruction = "I want you act as a Prompt Creator.\r\n\
Your goal is to draw inspiration from the #Given Prompt# to create a brand new prompt.\r\n\
This new prompt should belong to the same domain as the #Given Prompt# but be even more rare.\r\n\
The LENGTH and complexity of the #Created Prompt# should be similar to that of the #Given Prompt#.\r\n\
The #Created Prompt# must be reasonable and must be understood and responded by humans.\r\n\
'#Given Prompt#', '#Created Prompt#', 'given prompt' and 'created prompt' are not allowed to appear in #Created Prompt#\r\n"


class EvolutionTemplate:
    @staticmethod
    def generate_constraints_prompt(instruction):
        prompt = depth_base_instruction.format(
            "Please add one more constraints/requirements into #The Given Prompt#'"
        )
        prompt += "#The Given Prompt#: \r\n {} \r\n".format(instruction)
        prompt += "#Rewritten Prompt#:\r\n"
        return prompt

    @staticmethod
    def generate_deepen_prompt(instruction):
        prompt = depth_base_instruction.format(
            "If #The Given Prompt# contains inquiries about certain issues, the depth and breadth of the inquiry can be increased."
        )
        prompt += "#The Given Prompt#: \r\n {} \r\n".format(instruction)
        prompt += "#Rewritten Prompt#:\r\n"
        return prompt

    @staticmethod
    def generate_concretizing_prompt(instruction):
        prompt = depth_base_instruction.format(
            "Please replace general concepts with more specific concepts."
        )
        prompt += "#The Given Prompt#: \r\n {} \r\n".format(instruction)
        prompt += "#Rewritten Prompt#:\r\n"
        return prompt

    @staticmethod
    def generate_reasoning_prompt(instruction):
        prompt = depth_base_instruction.format(
            "If #The Given Prompt# can be solved with just a few simple thinking processes, you can rewrite it to explicitly request multiple-step reasoning."
        )
        prompt += "#The Given Prompt#: \r\n {} \r\n".format(instruction)
        prompt += "#Rewritten Prompt#:\r\n"
        return prompt

    @staticmethod
    def generate_breadth_prompt(instruction):
        prompt = breadth_base_instruction
        prompt += "#Given Prompt#: \r\n {} \r\n".format(instruction)
        prompt += "#Created Prompt#:\r\n"
        return prompt
