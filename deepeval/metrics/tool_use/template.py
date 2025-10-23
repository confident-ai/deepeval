import textwrap
import json


class ToolUseTemplate:

    @staticmethod
    def get_tool_selection_score(
        user_input: str,
        assistant_messages: str,
        tools_called: str,
        available_tools: str,
    ) -> str:
        return textwrap.dedent(
            f"""You are an expert evaluator assessing the **Tool Selection Quality** of an AI agent.

                OBJECTIVE
                Evaluate whether the agent **selected the most appropriate tools** for completing the user's task, given a list of available tools.

                This metric focuses **only** on which tools were chosen — **not** how they were used or whether they succeeded.

                EVALUATION RULES

                1. Relevance
                - Each tool used must directly support the user's stated goal or a clear sub-task derived from it.
                - Tools unrelated to the goal lower the score sharply.

                2. Appropriateness
                - The chosen tools must match their described purpose.
                - If a more suitable tool existed and was ignored, score ≤ 0.5.

                3. Necessity
                - Every tool call must be justified by clear need.
                - Redundant or speculative tool use (e.g., calling multiple tools that overlap) reduces the score.

                4. Strictness
                - When uncertain if a tool was required or correctly chosen, assume it was **not** appropriate.
                - Only perfect alignment between the task and tool choice earns a high score.

                SCORING GUIDE:

                - **1.0** → Every tool used was necessary and perfectly matched to the task; no better alternative ignored.  
                - **0.75** → Tool selection was mostly correct, with only minor redundancy or a small omission.  
                - **0.5** → Mixed quality; some appropriate selections, but others questionable or missing.  
                - **0.25** → Poor selection; major mismatches or misuse of available tools.  
                - **0.0** → Tool selection irrelevant, random, or unjustified.

                OUTPUT FORMAT:

                Return a JSON object with:
                
                {{
                    "score": float between 0.0 and 1.0,
                    "reason": "1-3 factual sentences explaining which tools were appropriate or inappropriate for the task, referencing specific tool names."
                }}

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
    def get_argument_correctness_score(
        user_input: str,
        assistant_messages: str,
        tools_called: str,
        available_tools: str,
    ) -> str:
        return textwrap.dedent(
            f"""You are an expert evaluator assessing the **Tool Argument Quality** of an AI agent.

                OBJECTIVE:

                Evaluate whether the **arguments and parameters** passed to each tool were:
                - Correctly structured and complete.
                - Contextually appropriate for the user's goal.
                - Compatible with each tool's intended purpose.

                This metric focuses **only** on argument-level correctness and relevance — not which tools were chosen.

                EVALUATION RULES

                1. Relevance
                - Each argument must align with the task and the tool's documented input fields.
                - Unrelated, empty, or default arguments reduce the score sharply.

                2. **Completeness**
                - All required parameters must be provided.
                - Missing or malformed arguments (e.g., wrong data types or incomplete context) lower the score.

                3. **Specificity**
                - Arguments should reflect task-specific values, not generic placeholders.
                - Overly vague or default arguments are penalized.

                4. **Justification**
                - Each argument must make sense in context.
                - If it doesn't clearly derive from the user's request, assume it's incorrect.

                5. **Strict Bias**
                - When uncertain whether arguments fit the tool or task, assume they were **incorrect**.

                SCORING GUIDE:

                - **1.0** → All arguments are accurate, specific, and fully aligned with both the task and tool requirements.  
                - **0.75** → Mostly correct; minor omissions or small mismatches.  
                - **0.5** → Partial correctness; some valid parameters, but key ones missing or off-target.  
                - **0.25** → Poor argument quality; several invalid or irrelevant fields.  
                - **0.0** → Arguments nonsensical, generic, or unrelated to task/tool intent.

                OUTPUT FORMAT:

                Return a JSON object with:
                {{
                    "score": float between 0.0 and 1.0,
                    "reason": "1-3 sentences explaining argument alignment or issues, referencing specific parameter names or values when possible."
                }}

                ---

                USER INPUT:
                {user_input}

                ASSISTANT MESSAGES:
                {assistant_messages}

                TOOLS CALLED (with arguments):
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
            f"""You are an expert evaluator summarizing the outcome of a **Tool Selection** evaluation.

            You are given:
            - A list of **tool selection sub-scores and reasons**, each describing how appropriately the agent chose tools for its task.
            - The **final aggregated score** across all sub-evaluations.
            - A **threshold** representing the minimum passing score.

            Your task is to write a **single concise explanation (1-3 sentences)** that captures:
            - Why the agent **passed or failed** based on tool choice quality.
            - The key patterns or trends in the sub-reasons (e.g., consistent correct choices, repeated irrelevant tool calls, missed best-fit tools).
            - A clear statement linking the **score** and **threshold** outcome (e.g., “The agent passed because…” or “Failed because…”).

            RULES:
            - Focus on *which tools were selected* and *why that selection pattern was or wasn't appropriate*.
            - Mention specific issues or strengths like redundancy, misuse, or perfect matching.
            - Avoid vague or subjective language such as “pretty good” or “reasonable”.
            - Do **not** reference argument-level details; this summary is only for tool choice quality.
            - The result must read as a self-contained, factual justification.

            FORMAT:
            Return only a single plain-text string. Do **not** include JSON or other formatting.

            All Tool Selection Sub-Scores and Reasons:
            {all_scores_and_reasons}

            Final Score: {final_score}
            Threshold: {threshold}
            Result: {"PASS" if final_score >= threshold else "FAIL"}

            Final Reason:
            """
        )

    @staticmethod
    def get_tool_argument_final_reason(
        all_scores_and_reasons: str, final_score: float, threshold: float
    ) -> str:
        return textwrap.dedent(
            f"""You are an expert evaluator summarizing the outcome of a **Tool Argument Quality** evaluation.

            You are given:
            - A list of **argument-level sub-scores and reasons**, each evaluating whether the arguments passed to tools were accurate, complete, and contextually appropriate.
            - The **final aggregated score** across all argument evaluations.
            - A **threshold** representing the minimum passing score.

            Your task is to write a **single concise explanation (1-3 sentences)** that clearly states:
            - Why the agent **passed or failed** in its use of tool arguments.
            - The dominant strengths or weaknesses from the sub-reasons (e.g., correct parameterization, missing required fields, generic values, or misaligned arguments).
            - Whether the agent met or fell short of the threshold and why.

            RULES:
            - Focus strictly on **argument correctness** and **context alignment** — not which tools were chosen.
            - Reference specific argument-level problems or successes where helpful.
            - Keep language objective and factual; avoid speculation or vague phrasing.
            - The summary must stand alone as a clear explanation of the final result.

            FORMAT:
            Return only a single plain-text string. Do **not** include JSON or any extra formatting.

            All Tool Argument Sub-Scores and Reasons:
            {all_scores_and_reasons}

            Final Score: {final_score}
            Threshold: {threshold}
            Result: {"PASS" if final_score >= threshold else "FAIL"}

            Final Reason:
            """
        )
