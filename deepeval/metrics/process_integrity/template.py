from typing import List


class ProcessIntegrityTemplate:

    @staticmethod
    def extract_conclusion(step: str) -> str:
        return f"""You are evaluating a single step from an AI agent's reasoning trajectory.

Your task: extract the logical conclusion this step reaches, and determine whether it is substantive reasoning or a non-substantive action (e.g., a raw tool call, API invocation, or mechanical action with no reasoning).

Step:
{step}

Respond ONLY with a JSON object. No explanation, no markdown, no preamble.

{{
  "conclusion": "<the logical conclusion this step reaches, or empty string if none>",
  "is_substantive": <true if this step contains reasoning or a conclusion, false if it is purely a tool call or mechanical action>
}}"""

    @staticmethod
    def check_integrity(
        step: str,
        conclusion: str,
        prior_conclusions: List[dict],
    ) -> str:
        prior_block = (
            "\n".join(
                f"Step {p['step_index']}: {p['conclusion']}"
                for p in prior_conclusions
            )
            if prior_conclusions
            else "None — this is the first substantive step."
        )

        return f"""You are evaluating logical consistency in an AI agent's reasoning trajectory.

Current step:
{step}

Conclusion reached in this step:
{conclusion}

Prior conclusions from earlier steps:
{prior_block}

Rules:
- If there are no prior conclusions, always return PASS.
- Return FAIL if this step's conclusion directly contradicts a prior conclusion without acknowledging the change.
- Return FAIL if this step skips required intermediate logic that should have appeared between the prior conclusion and this one.
- Return FAIL if this step inserts a conclusion that is irrelevant to the trajectory's reasoning chain.
- Return PASS otherwise.

Respond ONLY with a JSON object. No explanation, no markdown, no preamble.

If PASS:
{{
  "verdict": "PASS",
  "reason": "<brief explanation>"
}}

If FAIL:
{{
  "verdict": "FAIL",
  "reason": "<specific explanation of the contradiction, skip, or irrelevance>",
  "severity": "<low | medium | high>"
}}"""
