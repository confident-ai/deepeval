class ScorerTemplate:
    @staticmethod
    def generate_diagnosis(
        original_prompt: str,
        evaluation_results: str,
    ) -> str:
        return f"""You are an expert Prompt Engineer and AI Diagnoser. Your task is to perform a 'Prompt Gradient Analysis'. 

You are provided with:
1. The Original Prompt.
2. Evaluation Results: A batch of execution traces including Inputs, Expected Outputs, Actual Outputs, and Numerical Scores (0.0 to 1.0).

# Original Prompt:
'{original_prompt}'

# Evaluation Results
{evaluation_results}

# Instructions
Perform a precise diagnosis to guide the next mutation:
1. **Identify the High-Loss Examples:** Look for instances with the lowest numerical scores. Analyze exactly what caused the model to deviate from the expected output in these specific cases.
2. **Identify the Anchors:** Look for instances with scores of 1.0. Determine which parts of the prompt are working correctly so they aren't accidentally removed.
3. **Correlate Scores to Instructions:** Explicitly state: "Instruction X led to a score of 0.0 on Input Y because [reason]."
4. **Synthesize the 'Gradient'**: Provide a clear signal on what needs to be 'intensified' (added) or 'dampened' (removed/changed).

**Output Format**
Return a JSON object:
- "failures": List of failures.
- "successes": List of successes.
- "analysis": A synthesized diagnostic signal. You MUST include the numerical scores in your citations (e.g., "On example A (Score: 0.2), the model failed to...") to provide a magnitude of failure.

Example JSON:
{{
    "failures": ["The model consistently fails logic tests (Score 0.0) while passing formatting tests (Score 1.0)..."],
    "successes": ["The JSON formatting instruction is perfect (Score 1.0)."],
    "analysis": "CRITICAL FAILURE: The prompt lacks a 'step-by-step' requirement, leading to a 0.0 score on logic examples like [Quote]. SUCCESS: The JSON formatting instruction is perfect (Score 1.0)."
}}

JSON:
"""