from __future__ import annotations


class JudgeLMTemplate:
    """Prompt templates for LLM-as-a-judge pairwise test run comparisons."""

    DEFAULT_CRITERIA = (
        "Evaluate which response is superior based on factual accuracy, "
        "relevance to the input, completeness, coherence, and adherence to instructions. "
        "If expected output is provided, prioritize alignment with expected output. "
        "If both responses are of equal quality, or differences are negligible, declare a tie."
    )

    @staticmethod
    def compare_outputs(
        input_text: str,
        baseline_output: str | None,
        candidate_output: str | None,
        expected_output: str | None = None,
        context: list[str] | None = None,
        criteria: str | None = None,
    ) -> str:
        eval_criteria = criteria or JudgeLMTemplate.DEFAULT_CRITERIA

        context_section = ""
        if context:
            formatted_context = "\n".join(f"- {c}" for c in context)
            context_section = f"\nContext:\n{formatted_context}\n"

        expected_section = ""
        if expected_output:
            expected_section = (
                f"\nExpected Output (Reference):\n{expected_output}\n"
            )

        baseline_str = (
            baseline_output if baseline_output is not None else "[No Output]"
        )
        candidate_str = (
            candidate_output if candidate_output is not None else "[No Output]"
        )

        return f"""You are an expert impartial judge evaluating the responses from two different model runs (Baseline vs. Candidate) on the same test input.

Evaluation Criteria:
{eval_criteria}
{context_section}{expected_section}
Input Prompt:
{input_text}

---
[Response A (Baseline)]:
{baseline_str}

---
[Response B (Candidate)]:
{candidate_str}

---
Instructions:
1. Compare Response A (Baseline) and Response B (Candidate) objectively against the input prompt and criteria.
2. Determine which response is better:
   - Choose "candidate" if Response B (Candidate) is better.
   - Choose "baseline" if Response A (Baseline) is better.
   - Choose "tie" if both responses are equally good, equally flawed, or indistinguishable in quality.
3. Provide a clear, concise justification for your decision in the "reason" field.
4. Output strictly valid JSON matching the schema below. Do NOT wrap in markdown fences or include commentary outside the JSON.

Schema:
{{
  "winner": "candidate" | "baseline" | "tie",
  "reason": "<concise justification>"
}}

JSON:
"""
