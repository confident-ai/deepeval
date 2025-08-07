from deepeval.test_case import LLMTestCase


class ToolCorrectnessTemplate:
    @staticmethod
    def get_tool_correctness_prompt(
        test_case: LLMTestCase,
        available_primitives: str,
        primitives_used: str,
    ):
        return f"""Evaluate whether the tools (primitives) selected and used by the agent were appropriate and correct for fulfilling the user’s request. Base your judgment on the user input, the agent’s visible output, and the tools that were available to the agent. You must return a JSON object with exactly two fields: 'score' and 'reason'.

Scoring:
- 'score' is a float between 0 and 1 inclusive.
- Use intermediate values (e.g., 0.25, 0.5, 0.75) to reflect cases where the tools used were partially correct, suboptimal, or only somewhat relevant.
- 'reason' should clearly explain how appropriate and correct the chosen primitives were, considering both the user's request and the output.

IMPORTANT:
- Focus only on tool selection and usage — not the quality of the final output.
- Assume that 'available_primitives' contains the only tools the agent could have used.
- Consider whether the agent:
  - Chose the correct tool(s) for the task.
  - Avoided unnecessary or incorrect tool calls.
  - Missed a more appropriate tool when one was available.
- Multiple valid tool combinations may exist — give credit when one reasonable strategy is used effectively.

CHAIN OF THOUGHT:
1. Determine what the user was asking for from 'test_case.input'.
2. Evaluate whether the tools in 'primitives_used' were appropriate for achieving that goal.
3. Consider the list of 'available_primitives' to judge if better options were missed or if poor tools were unnecessarily used.
4. Ignore whether the tool *worked* — focus only on whether it was the *right tool to use*.

You must return only a valid JSON object. Do not include any explanation or text outside the JSON.

-----------------
User Input:
{test_case.input}

Agent Visible Output:
{test_case.actual_output}

Available Tools:
{available_primitives}

Tools Used by Agent:
{primitives_used}

Example Output:
{{
    "score": 0.75,
    "reason": "The agent used a relevant tool to address the user's request, but a more specific tool was available and would have been more efficient."
}}

JSON:
"""
