class COPROTemplate:

    @staticmethod
    def generate_bootstrap_guidelines(
        original_prompt: str, breadth: int
    ) -> str:
        return f"""You are an expert prompt engineer. I need to generate {breadth} distinct, high-quality variations of the following prompt.

[ORIGINAL PROMPT]
{original_prompt}

[INSTRUCTIONS]
Brainstorm exactly {breadth} diverse "Variation Guidelines". Each guideline should be a 1-2 sentence strategy on how to significantly alter or improve the prompt (e.g., changing the tone, adding reasoning steps, enforcing specific output formats, reordering instructions). 
Make sure the guidelines are completely distinct from one another to ensure a wide search space.

**
IMPORTANT: You must only return in JSON format matching the schema.
Example JSON:
{{
    "guidelines": [
        "Reframe the prompt to require step-by-step chain of thought before providing the final answer.",
        "Condense the instructions into a highly aggressive, concise format avoiding any pleasantries.",
        "Add strict formatting constraints requiring the output to be bulleted."
    ]
}}
**

JSON:
"""

    @staticmethod
    def generate_history_guidelines(
        original_prompt: str, history_text: str, breadth: int
    ) -> str:
        return f"""You are an expert prompt engineer and diagnostic system. We are using Coordinate Ascent to optimize a prompt. 

[ORIGINAL PROMPT]
{original_prompt}

[PAST ATTEMPTS, SCORES, & EVALUATION FEEDBACK]
{history_text}

[INSTRUCTIONS]
Analyze the [PAST ATTEMPTS, SCORES, & EVALUATION FEEDBACK]. Higher scores are better. 
Crucially, look at the "Evaluation Feedback" for each attempt. This tells you exactly why the prompt lost points (e.g., failed a toxicity metric, missed a formatting constraint).

Based on this analysis, brainstorm exactly {breadth} new "Variation Guidelines" to try next. 
These guidelines MUST explicitly address and fix the errors mentioned in the evaluation feedback while maintaining the successful traits of the highest-scoring prompts.

**
IMPORTANT: You must only return in JSON format matching the schema.
Example JSON:
{{
    "guidelines": [
        "The highest scoring prompts used step-by-step reasoning, but failed the JSON format metric. Add a strict JSON schema constraint.",
        "Past attempts failed the toxicity metric when being too aggressive. Create a variation that is highly polite but retains the reasoning steps."
    ]
}}
**

JSON:
"""

    @staticmethod
    def generate_candidate(
        original_prompt: str, guideline: str, is_list_format: bool = False
    ) -> str:

        # Dynamically instruct the LLM on how to format the revised_prompt field
        if is_list_format:
            format_instruction = (
                "A JSON array of message objects representing the revised conversational prompt "
                '(e.g., [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]).'
            )
            example_instruction = '[\n        {"role": "system", "content": "You are a helpful assistant."},\n        {"role": "user", "content": "{{input}}"}\n    ]'
        else:
            format_instruction = (
                "The final string representing the optimized revised prompt."
            )
            example_instruction = (
                '"You are a helpful assistant. Please answer: {{input}}"'
            )

        return f"""You are an expert prompt engineer. Your task is to rewrite a prompt based strictly on a specific optimization guideline.

[ORIGINAL PROMPT]
{original_prompt}

[OPTIMIZATION GUIDELINE]
{guideline}

[INSTRUCTIONS]
Rewrite the [ORIGINAL PROMPT] applying the [OPTIMIZATION GUIDELINE]. 
1. The new prompt must fulfill the core task of the original prompt.
2. DO NOT wrap your revised_prompt in markdown blocks (like ```).
3. If the original prompt uses variable placeholders (like {{input}}), you MUST retain them.

**
IMPORTANT: You must only return in JSON format matching the schema.
"revised_prompt" format: {format_instruction}

Example JSON:
{{
    "thought_process": "The guideline asks to make the prompt more concise. I will remove the introductory pleasantries and state the objective directly.",
    "revised_prompt": {example_instruction}
}}
**

JSON:
"""
