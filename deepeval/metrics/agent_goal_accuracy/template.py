from typing import List
import textwrap


class AgentGoalAccuracyTemplate:
    @staticmethod
    def get_accuracy_score(task, steps_taken):
        return textwrap.dedent(
            f"""You are an expert evaluator assessing the **goal accuracy** of an AI agent based on a given task and the steps the agent took to complete it.

                You are given:
                - A **user task**: the user's original request or objective, clearly describing what they wanted the agent to accomplish.
                - A list of **agent steps**: a complete trace of what the agent did to fulfill the task. This includes:
                - All **messages** the agent sent to the user (which the user sees).
                - Any **tool calls** or intermediate actions (which the user does not see unless results are explicitly stated in agent messages).

                Your job is to determine how completely and correctly the agent fulfilled the task **from the user's perspective** — based only on what the user would experience.

                ---

                SCORING RULES:

                You must return a **JSON object** with exactly two fields:
                - `"score"`: a float from 0.0 to 1.0 representing how accurately the task was completed.
                - `"reason"`: 1-3 sentences explaining your score, focusing only on what was visible to the user.

                Use this scoring guide:
                - **1.0** → Task fully and correctly completed; all expected outputs were clearly delivered to the user.
                - **0.75** → Mostly completed; some minor outputs were missing, ambiguous, or slightly inaccurate.
                - **0.5** → Partially completed; major parts of the task were fulfilled, but key steps were missing or incorrect.
                - **0.25** → Minimally completed; some attempt was made, but the result would feel incomplete or wrong to the user.
                - **0.0** → Task not completed at all; the user would receive no useful or relevant output.

                ---

                IMPORTANT RULES:

                - Do **not** consider internal tool calls or retrievals unless the results were surfaced in an agent message.
                - Assume the **user only sees the assistant's replies** — no internal state, tool inputs, or outputs.
                - Penalize any visible **inaccuracies, hallucinations, omissions**, or **failures to report tool results**.
                - The score must be based on **user-visible experience only**.

                ---

                CHAIN OF THOUGHT:

                1. Break the user task into distinct actionable parts.
                2. Review the agent's replies to see which parts were clearly and correctly completed.
                3. Check whether any tool outputs were mentioned to the user — if not, they do not count toward goal completion.
                4. Determine if the agent's final response leaves the user with a complete, correct, and self-contained answer.

                ---

                FORMAT:

                Return only a valid JSON object — no commentary or extra text.

                Example:

                {{
                "score": 0.5,
                "reason": "The agent responded with a partial itinerary but failed to include hotel options, and tool results were not mentioned to the user."
                }}

                ------------------

                User Task:
                {task}

                Agent Steps:
                {steps_taken}

                JSON:
            """
        )
    
    @staticmethod
    def get_plan_evaluation_score(task, steps_taken):
        return textwrap.dedent(
            f"""You are an expert evaluator assessing the **planning quality** and **plan adherence** of an AI agent.

                You are given:
                - A **user task**: a clear description of what the user asked the agent to do.
                - A list of **agent steps**: all visible actions and responses made by the agent, including any tool calls and final replies.

                Your job is to analyze:
                1. **Plan Quality**: Did the agent demonstrate a well-structured, logical plan to fulfill the user's task?
                2. **Plan Adherence**: Did the agent's execution follow through on that plan without unnecessary deviations or omissions?

                ---

                SCORING FORMAT:

                You must return a JSON object with exactly two fields:
                - `"score"`: a float from 0.0 to 1.0 measuring the overall strength of the agent's planning and adherence.
                - `"reason"`: a concise explanation (1-3 sentences) justifying the score.

                Use intermediate values (e.g. 0.25, 0.5, 0.75) to reflect partial planning or minor deviations.

                ---

                SCORING GUIDE:

                - **1.0** → The agent clearly formed a complete, logical plan and followed it closely. All steps were purposeful and aligned with the user's goal.
                - **0.75** → The plan was mostly good, with minor issues or small detours in execution that did not materially impact the outcome.
                - **0.5** → The plan had some valid components but was incomplete, vague, or partially abandoned during execution.
                - **0.25** → The agent showed limited or unclear planning, and execution drifted from any coherent strategy.
                - **0.0** → No evidence of a meaningful plan; actions were random, reactive, or unrelated to the user's goal.

                ---

                CHAIN OF THOUGHT:

                1. Identify whether the agent had an implicit or explicit **plan** to complete the user's task.
                2. Determine whether this plan was logically sound and sufficiently detailed.
                3. Compare the agent's **execution steps** against this plan:
                    - Were they consistent with the plan?
                    - Did they skip or add steps without justification?
                4. Penalize planning that was vague, overly generic, or clearly ignored during execution.

                ---

                IMPORTANT:

                - This evaluation does **not** judge correctness or efficiency — it only evaluates **planning and execution alignment**.
                - If the agent did not show a clear plan, score should be **≤ 0.5**, even if the task was eventually completed.
                - Tool use is only valuable if it fits into a well-reasoned and visible plan.

                ---

                FORMAT:

                Return only a valid JSON object — no commentary or extra text.

                Example:

                {{
                    "score": 0.75,
                    "reason": "The agent demonstrated a clear plan to gather flights and hotels, but skipped mentioning the slide deck retrieval despite having included it in the early structure."
                }}

                ------------------

                User Task:
                {task}

                Agent Steps:
                {steps_taken}

                JSON:
            """
        )
    
    @staticmethod
    def get_final_reason(final_score, threshold, goal_evaluations, plan_evalautions):
        return textwrap.dedent(
            f"""You are an expert evaluator providing a **final justification** for whether an AI agent has passed or failed an evaluation metric.

                You are given:
                - An agent's goal execution scores and reasons.
                - The agent's plan evaluation scores and reasons.
                - The **final combined score**.
                - The **threshold** required to pass.
                - Whether the result is a **pass** or **fail**.

                Your job is to write a short, precise explanation of **why** the agent passed or failed — taking into account the quality of execution and planning, and the threshold.

                ---

                INSTRUCTIONS:

                - Write 2-4 clear, objective sentences explaining the overall result.
                - Explicitly reference both the task and plan performance — **both must be addressed**.
                - Mention how the final score compares to the threshold.
                - If the agent **passed**, highlight how both task execution and planning were sufficient to meet the goal.
                - If the agent **failed**, explain which aspects (task or plan or both) led to the failure.
                - Avoid vague praise or criticism — ground the reason in the actual scores and justifications.

                ---

                FORMAT:
                Return only a single string. Do **not** include JSON or any extra formatting.

                ---

                Goal evaluations:
                {goal_evaluations}

                Plan evaluations:
                {plan_evalautions}

                Final Score: {final_score}
                Threshold: {threshold}
                Result: {"PASS" if final_score >= threshold else "FAIL"}

                Final Reason:
            """
        )
