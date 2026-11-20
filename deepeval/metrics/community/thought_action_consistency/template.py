import json
from typing import List, Tuple

from deepeval.test_case import ToolCall


class ThoughtActionConsistencyTemplate:
    """Prompt for per-step thought/action consistency.

    Unlike ToolCorrectnessMetric (which compares executed tool calls against a
    golden ``expected_tools`` list) and StepEfficiencyMetric (which asks a judge
    for one holistic efficiency score over the trace), this prompt asks the
    judge, for each tool call, a narrow internal-consistency question: does the
    executed call — tool choice and arguments — execute the intent the
    reasoning step declared for it? It deliberately does NOT ask whether the
    step was the best possible move, and does NOT compare against any external
    expectation.
    """

    @staticmethod
    def format_steps(
        judged_steps: List[Tuple[int, ToolCall]],
    ) -> str:
        """Render each judged step with its original index in the trace.

        ``judged_steps`` is a list of ``(original_index, call)`` pairs where
        ``original_index`` is the 1-based position of the call in the full
        ``tools_called`` list. Keeping the original index means the judge's
        verdicts and the metric's reason map back to the trace the user sees,
        even when calls without reasoning were skipped.
        """
        lines = []
        for original_index, call in judged_steps:
            lines.append(
                f"Step {original_index} — Reasoning: {call.reasoning}\n"
                f"         Tool: {call.name}\n"
                f"         Input parameters: "
                f"{json.dumps(call.input_parameters or {}, ensure_ascii=False, default=str)}"
            )
        return "\n\n".join(lines)

    @staticmethod
    def generate_verdict(steps: str) -> str:
        return f"""You are auditing the internal consistency of a ReAct-style agent's execution trace.
For each numbered step you are given the agent's reasoning (its stated intent for that step), the tool it called, and the arguments it passed.
For each step, decide whether the executed call CONSISTENTLY carries out the intent the reasoning declared.

Definitions:
- consistent: the tool called is a reasonable direct execution of the intent stated in the reasoning, and the arguments match the specifics the reasoning gave.
- inconsistent, in one of exactly three ways:
  1. tool mismatch: the reasoning declares one kind of action but a different tool was called (e.g. reasoning says "query the database logs" but a metrics API is called);
  2. argument mismatch: the tool is right but an argument contradicts or ignores the specifics the reasoning stated (e.g. reasoning says "check the payment service" but the service argument is "auth-api");
  3. unmotivated call: the reasoning declares no action, yet a tool call happened.

Scope rules — judge ONLY the internal consistency of each step:
- Do NOT judge whether the step was the best or most efficient way to progress the task (a different metric covers efficiency).
- Do NOT judge whether the tool's result was correct or useful.
- Do NOT compare the trace against any external expected plan or golden sequence.

Return a JSON object with a single key "verdicts": a list with one entry per judged step, in step order. Each entry has "step_index" (the number shown next to that step below), "verdict" ("consistent" or "inconsistent"), and "reasoning" (one sentence citing which way, if any, the step mismatched).

Example JSON:
{{
    "verdicts": [
        {{"step_index": 1, "verdict": "consistent", "reasoning": "The call queries the log database with the service the reasoning named."}},
        {{"step_index": 2, "verdict": "inconsistent", "reasoning": "Argument mismatch: the reasoning says to check the payment service but the service argument is auth-api."}}
    ]
}}

Steps:
{steps}

JSON:
"""

    @staticmethod
    def _experimental_system_one_verdict() -> str:
        return (
            "Is every judged tool call consistent with the reasoning that "
            "preceded it? Answer yes only if, for every step in `steps` that "
            "carries a reasoning, the tool called and its arguments directly "
            "execute the intent the reasoning declared. Answer no if any step "
            "shows a tool mismatch, an argument mismatch, or an unmotivated "
            "call. Judge only internal consistency; do not assess efficiency "
            "or result correctness."
        )

    @staticmethod
    def _experimental_system_one_questions() -> str:
        return """[
  {
    "type": "noul",
    "statement": "Every judged step in `steps` executes the intent its reasoning declared.",
    "weight": 2
  },
  {
    "type": "score",
    "question": "What fraction of the judged steps are consistent with the reasoning that preceded them?",
    "levels": ["None of them", "Less than half", "About half", "Most of them", "All of them"]
  }
]"""
