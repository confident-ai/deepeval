from typing import List, Dict
from deepeval.metrics.mcp.schema import Task
from deepeval.test_case import MCPServer


class MCPTaskCompletionTemplate:
    @staticmethod
    def get_args_correctness_score(task: Task, mcp_servers: List[MCPServer]):
        available_tools = [data.available_tools for data in mcp_servers]
        available_resources = [data.available_resources for data in mcp_servers]
        available_prompts = [data.available_prompts for data in mcp_servers]
        steps_taken = "\n".join(task.steps_taken)
        return f"""Evaluate whether the arguments (inputs) provided by the agent to the tools, resources, and prompts were correct and aligned with their respective input schemas. Your job is to determine if the agent supplied appropriate, complete, and well-formatted arguments for each invocation.

Output a JSON object with exactly two fields: 'score' and 'reason'.

Scoring:
- 'score' is a float between 0 and 1 inclusive.
- Use intermediate values (e.g., 0.25, 0.5, 0.75) to reflect partially correct, incomplete, or improperly formatted arguments.
- 'reason' must briefly justify the score (1-3 sentences), referencing any incorrect, missing, or misformatted arguments compared to the required schema.

CHAIN OF THOUGHT:
1. Review each step where a tool, resource, or prompt was called.
2. Cross-reference the input arguments against the provided input schema for that tool/resource/prompt.
3. Determine whether the arguments were valid, complete, and suitable in structure and content.
4. Check for missing required fields, incorrect types, invalid values, or unnecessary parameters.
5. Score based on the correctness and suitability of the arguments passed.

Return only a valid JSON object. Do not include any explanation or text outside the JSON.

-----------------
User Task:
{task.task}

Input Schemas:
{available_tools}\n
{available_resources}\n
{available_prompts}\n

Agent Steps:
{steps_taken}

Example Output:
{{
  "score": 0.5,
  "reason": "The agent provided mostly valid fields, but omitted a required parameter and used a string where a list was expected."
}}

JSON:
"""

    @staticmethod
    def get_tool_correctness_score(task: Task, mcp_servers: List[MCPServer]):
        available_tools = [data.available_tools for data in mcp_servers]
        steps_taken = "\n".join(task.steps_taken)
        return f"""Evaluate whether the tools, resources, and prompts used by the agent were appropriate and optimal, based strictly on the list of available tools and resources provided. Your job is to determine whether the agent selected the most suitable tools and prompts for the task at hand. Output a JSON object with exactly two fields: 'score' and 'reason'.

Scoring:
- 'score' is a float between 0 and 1 inclusive.
- Use intermediate values (e.g., 0.25, 0.5, 0.75) to reflect partially appropriate tool use, suboptimal decisions, or missed better alternatives.
- 'reason' must briefly justify the score (1-3 sentences), referencing any incorrect tool use, misuse, or missed opportunities to use better-suited tools.

CHAIN OF THOUGHT:
1. Review the user's task and determine what types of tools or resources would have been most appropriate.
2. Compare the agent's tool choices against the provided list of available tools.
3. Verify whether any better-suited tools or resources were omitted.
4. Check for any misuse or unnecessary use of tools or resources.
5. Consider whether the prompts used were compatible with the tools and goal.

Return only a valid JSON object. Do not include any explanation or text outside the JSON.

-----------------
User Task:
{task.task}

Available Tools:
{available_tools}

Agent Steps:
{steps_taken}

Example Output:
{{
  "score": 0.75,
  "reason": "The agent used a tool that was generally appropriate but missed a more specialized tool available in the list that could have provided more accurate results."
}}

JSON:
"""

    @staticmethod
    def get_task_completion_score(task: Task):
        steps_taken = "\n".join(task.steps_taken)
        return f"""Evaluate whether the user's task has been successfully completed by the agent, based strictly on what the user can see in the agent's responses. You must return a JSON object with exactly two fields: 'score' and 'reason'.
Scoring:
- 'score' is a float between 0 and 1 inclusive.
- Use intermediate values (e.g., 0.25, 0.5, 0.75) to reflect partial task success or missing/inaccurate information.
- 'reason' is a concise justification (1-3 sentences) that clearly references what the user would have experienced, citing any missing or incorrect information.

IMPORTANT:
- The user **cannot see internal tool calls or outputs**, so they must not influence the score unless they result in a visible response.
- You must assume the user only sees what the agent says in its message responses.

CHAIN OF THOUGHT:
1. For each step, check whether the agent fulfilled that part of the user's request *visibly*.
2. Confirm that any claims made by the agent (e.g. “I did the following”) are *actually supported* by what was displayed.
3. Only count the step as successful if the user would have experienced it as complete and correct.

You must return only a valid JSON object. Do not include any explanation or text outside the JSON.

-----------------
User Task:
{task.task}

Agent Steps:
{steps_taken}

Example Output:
{{
    "score": 1.0,
    "reason": "The agent successfully completed all required steps with accurate results."
}}

JSON:
"""
