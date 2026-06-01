class RewriterTemplate:
    @staticmethod
    def generate_mutation(
        original_prompt: str,
        failures: str,
        successes: str,
        results: str,
        analysis: str,
        is_list_format: bool = False,
    ) -> str:

        if is_list_format:
            format_instruction = (
                "A JSON array of message objects representing the revised conversational prompt "
                "(e.g., [{'role': 'system', 'content': '...'}, {'role': 'user', 'content': '...'}])."
            )
            example_prompt = '[{"role": "system", "content": "You are a helpful assistant..."},{"role": "user", "content": "{{input}}"}]'
        else:
            format_instruction = (
                "The final string representing the optimized revised prompt."
            )
            example_prompt = '"<the optimized revised prompt here>"'

        return f"""You are an expert AI Prompt Engineer. Your goal is to perform a 'Prompt Mutation' to move the prompt closer to the Pareto Frontier.

# Context
- **Original Prompt:** The current best-performing candidate.
- **Diagnostic Report:** A 'gradient' signal identifying high-loss areas (low scores) and anchors (high scores).
- **Failure Cases:** The failure cases from the diagnostic report.
- **Success Cases:** The success cases from the diagnostic report.
- **Actual Results:** The actual results from the previous generation.
- **Overall Analysis:** The overall analysis of the diagnostic report.

# Original Prompt
{original_prompt}

# Diagnostic Report
Failures: {failures}
Successes: {successes}

Actual results from the previous generation: {results}

Overall analysis of the diagnostic report: {analysis}

# Mutation Instructions
1. **Targeted Fixes:** Use the Diagnostic Report to apply 'surgical' edits. Focus heavily on the examples that received low numerical scores.
2. **Constraint Satisfaction:** Do NOT degrade performance on the 'Anchor' examples (those with 1.0 scores). Your mutation must be a 'non-dominated' improvement.
3. **Preserve Placeholders:** Maintain all runtime tokens like `{{input}}` or `{{context}}`.
4. **Iterative Refinement:** If the report mentions a lack of clarity, add explicit 'Rules' or 'Negative Constraints' (what NOT to do).
5. Always keep the interpolation type of the prompt the same as the original prompt. We use regex to interpolate the prompt so keep the same format.

**Output Format**
Return a JSON object:
- "thought_process": Explain how you are addressing the low-score failures while preserving high-score successes.
- "revised_prompt": {format_instruction}

Example JSON:
{{
    "thought_process": "<your reasoning here>",
    "revised_prompt": {example_prompt}
}}


JSON:
"""
