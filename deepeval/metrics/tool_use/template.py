import textwrap
import json


class ToolUseTemplate:

    @staticmethod
    def get_tool_selection_score(
        user_input: str, assistant_messages: str, tools_called: str, available_tools: str
    ) -> str:
        return textwrap.dedent(
            f"""You are an expert evaluator assessing the **Tool Selection** quality of an AI agent.

            You are given:
            - The **user's input**, which defines their goal.
            - The **assistant's visible messages**, which reflect what the user saw.
            - A list of **tool calls made** by the agent, including names and parameters.
            - A list of **available tools**, each with a name and description.

            Your task is to assign a **Tool Selection score** between 0.0 and 1.0, along with a reasoned justification.

            ---

            DEFINITION:

            Tool Selection evaluates whether the tools selected by the agent were **appropriate and well-matched** to the tasks and sub-tasks implied by the user's input.

            It does **not** evaluate:
            - How well the tools were used (execution quality)
            - Whether the output was correct or efficient
            - Whether the agent adhered to a plan

            This metric **only** assesses whether the **right tools** were chosen based on their descriptions and the user's visible request.

            ---

            INSTRUCTIONS:

            Step 1: Read the **user_input** to infer the user's goal and any implied sub-tasks.

            Step 2: Review the **available_tools** and their descriptions to understand the purpose and boundaries of each tool.

            Step 3: Analyze the **tools_called**:
            - Were the selected tools appropriate for the user's goal?
            - Were any better-suited tools ignored?
            - Were any irrelevant or unnecessary tools called?
            - Were any tools misapplied (used for tasks outside their described purpose)?

            Step 4: Use the **assistant_messages** to verify which aspects of the goal the agent visibly attempted to solve — and whether the selected tools aligned with that.

            ---

            SCORING GUIDE:

            - **1.0** → All selected tools were clearly appropriate and necessary for the tasks. No better-suited tools were ignored.
            - **0.75** → Mostly correct selection, with only minor redundancy or omission.
            - **0.5** → Mixed selection quality. Some useful tools ignored or some questionable ones used.
            - **0.25** → Tool choice was poorly aligned. Better tools were available but not used.
            - **0.0** → Tool selection was fundamentally flawed or unjustified.

            ---

            OUTPUT FORMAT:

            Return a valid JSON object in the following format:
            {{
                "score": float between 0.0 and 1.0,
                "reason": "1-3 objective sentences that explain the score, referencing specific tool names and purposes where needed."
            }}

            Do not include any extra commentary, formatting, or explanation outside the JSON object.

            ---

            USER INPUT:
            {user_input}

            ASSISTANT MESSAGES:
            {assistant_messages}

            TOOLS CALLED:
            {tools_called}

            AVAILABLE TOOLS:
            {available_tools}

            JSON:
            """
        )

    @staticmethod
    def get_tool_selection_final_reason(
        all_scores_and_reasons: str, final_score: float, threshold: float
    ) -> str:
        return textwrap.dedent(
            f"""You are an expert evaluator summarizing the result of a **Tool Selection** evaluation.

            You are given:
            - A list of **tool selection sub-scores and their reasons**, each analyzing how appropriate the agent's tool choices were for the task.
            - The **final aggregated score** for tool selection.
            - The **threshold** required to pass.

            Your task is to write a **single, concise explanation** (1-3 sentences) of **why** the agent passed or failed this metric, based on the score and the evaluations.

            ---

            INSTRUCTIONS:

            - Reference key strengths or weaknesses from the sub-reasons — especially misuse, omissions, or strong alignments.
            - Mention whether the tool choices aligned well (or not) with the task and available tools.
            - Clearly indicate whether the final score met or fell below the threshold, and why.
            - Avoid vague language like "pretty good" or "seems fine".
            - The response should **stand alone** as a clear summary of the result.

            ---

            FORMAT:
            Return only a single plain-text string. Do **not** include JSON or any formatting.

            ---

            All Scores and Reasons:
            {all_scores_and_reasons}

            Final Score: {final_score}
            Threshold: {threshold}
            Result: {"PASS" if final_score >= threshold else "FAIL"}

            Final Reason:
            """
        )
