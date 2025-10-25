import textwrap
import json


class ToolCorrectnessTemplate:

    @staticmethod
    def get_tool_selection_score(
        user_input: str, tools_called: list, available_tools: list
    ) -> str:
        return textwrap.dedent(
            f"""You are an expert evaluator assessing the **Tool Selection** quality of an AI agent.

            You are given:
            - The **user input** that defines the user's goal / task.
            - A list of **available tools**, each with a name and description.
            - A list of **tool calls made** by the agent during execution, including tool name and parameters.

            Your job is to assign a **Tool Selection score** from 0.0 to 1.0 based on how appropriate and well-matched the agent's chosen tools were to the task's requirements.

            ---

            DEFINITION:

            Tool Selection evaluates how suitable the agent's tool choices were in addressing the task and sub-tasks.

            This metric does **not** consider:
            - How well the tools were used (execution quality)
            - Whether the agent adhered to a plan
            - Whether the output was correct or efficient

            It only assesses whether the **right tools** were selected, based on their stated descriptions and the demands of the task.

            ---

            INSTRUCTIONS:

            Step 1: Read the **user task** to understand what needed to be accomplished.

            Step 2: Examine the **available tools** and their descriptions to understand the intended purpose of each.

            Step 3: Review the **tool calls made by the agent**:
            - Were the selected tools well-aligned with the task?
            - Were any obviously better-suited tools ignored?
            - Were any tools misapplied or used unnecessarily?

            Step 4: Identify selection issues:
            - **Correct Selection**: Tool(s) chosen directly and appropriately matched the subtask.
            - **Over-selection**: More tools were selected than necessary, despite availability of a simpler or more direct option.
            - **Under-selection**: Key tools that were well-suited were omitted.
            - **Mis-selection**: Tools were chosen that were poorly matched to their purpose or the subtask.

            ---

            SCORING GUIDE:

            - **1.0** → All selected tools were appropriate and necessary. No better-suited tools were omitted.
            - **0.75** → Tool choices were mostly appropriate, with minor omissions or unnecessary use.
            - **0.5** → Mixed tool selection. Some useful tools ignored or some inappropriate ones used.
            - **0.25** → Poor tool selection. Better alternatives were available and ignored.
            - **0.0** → Tool selection was clearly misaligned with task requirements.

            ---

            OUTPUT FORMAT:

            Return a valid JSON object with this exact structure:
            {{
                "score": float between 0.0 and 1.0,
                "reason": "1-3 concise, factual sentences explaining the score. Reference specific tool names and descriptions when relevant."
            }}

            Do not include any additional commentary or output outside the JSON object.

            ---

            USER INPUT:
            {user_input}

            ALL AVAILABLE TOOLS:
            {available_tools}

            TOOL CALLS MADE BY AGENT:
            {tools_called}

            JSON:
            """
        )
