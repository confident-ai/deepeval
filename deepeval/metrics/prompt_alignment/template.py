from typing import List


class PromptAlignmentTemplate:
    @staticmethod
    def generate_verdicts(
        prompt_instructions: List[str], input: str, actual_output: str
    ):
        return f"""For the provided list of prompt instructions, determine whether each instruction has been followed in the LLM actual output.
Please generate a list of JSON with two keys: `verdict` and `reason`.
The 'verdict' key should STRICTLY be either a 'yes' or 'no'. Only answer 'yes' if the instruction COMPLETELY follows the instruction, and 'no' otherwise.
You should be EXTRA STRICT AND CAREFUL when giving a 'yes'.
The 'reason' is the reason for the verdict.
Provide a 'reason' ONLY if the answer is 'no'. 
The provided prompt instructions are the instructions to be followed in the prompt, which you have no acccess to.

**
IMPORTANT: Please make sure to only return in JSON format, with the 'verdicts' key mapping to a list of JSON objects.
Example input: What number is the stars of the sky?
Example actual output: HEY THERE! I think what you meant is "What is the number of stars in the sky", but unforunately I don't know the answer to it.
Example prompt instructions: ["Answer the input in a well-mannered fashion.", "Do not correct user of any grammatical errors.", "Respond in all upper case"]
Example JSON:
{{
    "verdicts": [
        {{
            "verdict": "yes"
        }},
        {{
            "verdict": "no",
            "reason": "The LLM corrected the user when the user used the wrong grammar in asking about the number of stars in the sky."
        }},
        {{
            "verdict": "no",
            "reason": "The LLM only made 'HEY THERE' uppercase, which does not follow the instruction of making everything uppercase completely."
        }}
    ]  
}}

Since you are going to generate a verdict for each instruction, the number of 'verdicts' SHOULD BE STRICTLY EQUAL to the number of prompt instructions.
**          

Prompt Instructions:
{prompt_instructions}

Input:
{input}

LLM Actual Output:
{actual_output}

JSON:
"""

    @staticmethod
    def generate_reason(
        unalignment_reasons: List[str],
        actual_output: str,
        input: str,
        score: int,
    ):
        return f"""Given the prompt alignment score, the reaons for unalignment found in the LLM actual output, the actual output, and input, provide a CONCISE reason for the score. Explain why it is not higher, but also why it is at its current score.
The unalignments represent prompt instructions that are not followed by the LLM in the actual output.
If there no unaligments, just say something positive with an upbeat encouraging tone (but don't overdo it otherwise it gets annoying).
Don't have to talk about whether the actual output is a good fit for the input, access ENTIRELY based on the unalignment reasons.

**
IMPORTANT: Please make sure to only return in JSON format, with the 'reason' key providing the reason.
Example JSON:
{{
    "reason": "The score is <prompt_alignment_score> because <your_reason>."
}}
**

Input:
{input}

LLM Actual Output:
{actual_output}

Prompt Alignment Score:
{score}

Reasons for unalignment:
{unalignment_reasons}

JSON:
"""
