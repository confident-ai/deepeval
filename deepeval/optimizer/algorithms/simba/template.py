class SIMBATemplate:

    @staticmethod
    def generate_introspection_rewrite(
        original_prompt: str,
        worse_trajectory: str,
        better_trajectory: str,
        is_list_format: bool = False,
    ) -> str:

        if is_list_format:
            format_instruction = (
                "A STRICT JSON array of message objects representing the fully rewritten conversational prompt "
                "(e.g., [{'role': 'system', 'content': '...'}, {'role': 'user', 'content': '...'}])."
            )
            example_instruction = '[{"role": "system", "content": "You are a highly precise analytical engine. Always map out variables step-by-step before calculating..."},{"role": "user", "content": "{{input}}"}]'
        else:
            format_instruction = (
                "The final string representing the fully rewritten prompt."
            )
            example_instruction = '"You are a highly precise analytical engine. Always map out variables step-by-step before calculating. Input: {{input}}"'

        return f"""You are the core Introspective Rewriter Engine for SIMBA (Stochastic Introspective Mini-Batch Ascent), operating within a world-class prompt optimization framework. 
SIMBA optimizes prompts by hunting for high-variance 'hard' examples, sampling multiple trajectories, and learning from the delta between successful and failed executions on the exact same inputs.

Your objective is to analyze a language model's execution traces (a success and a failure), diagnose the root cause of the failure, and holistically rewrite the original prompt to structurally prevent this failure in the future.

[ORIGINAL INSTRUCTIONS]
{original_prompt}

[WORSE TRAJECTORY (The Failure)]
{worse_trajectory}

[BETTER TRAJECTORY (The Success)]
{better_trajectory}

[INSTRUCTIONS]
Conduct a deep introspection of the provided trajectories to execute the SIMBA optimization:
1. In the "discussion" field, rigorously contrast the WORSE and BETTER trajectories. 
   - Identify the exact delta in logic, formatting, or constraints that led to the worse score.
   - Synthesize a universal rule or "cheat code" that guarantees the behavior seen in the BETTER trajectory.
2. In the "revised_prompt" field, REWRITE the entire [ORIGINAL INSTRUCTIONS] from the ground up.
   - Seamlessly weave your synthesized rule natively into the core instructions. Do not just append a lazy rule at the bottom.
   - Improve the overall clarity, constraint enforcement, and reasoning structure of the prompt.
   - You MUST retain any exact variable placeholders from the original prompt (e.g., {{input}} or {{context}}).

**
IMPORTANT: You must ONLY return valid JSON matching the schema below. Do not wrap your response in markdown blocks (like ```json).
"revised_prompt" format: {format_instruction}

Example JSON:
{{
    "discussion": "The worse trajectory jumped straight to calculating the final value, causing a hallucination. The better trajectory explicitly mapped out the variables first. The structural rule is to force variable extraction before math operations.",
    "revised_prompt": {example_instruction}
}}
**

JSON:
"""
