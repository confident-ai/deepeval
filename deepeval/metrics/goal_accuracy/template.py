from typing import List
import textwrap


class GoalAccuracyTemplate:
    @staticmethod
    def get_accuracy_score(task, steps_taken):
        return textwrap.dedent(
            f"""You are an expert evaluator assessing the **goal accuracy** of an AI assistant's single interaction.

                PURPOSE:

                Evaluate whether the assistant's **visible output** (what the user actually saw) **fully and correctly achieved the user's stated goal.  
                Ignore internal reasoning, hidden tool calls, or retriever outputs unless their results were explicitly surfaced to the user.

                The evaluation must be **strict and adversarial** — if the goal is not *clearly, fully, and correctly achieved*, assign a low score.

                EVALUATION RULES

                1. **User-visible fulfillment only**
                - Base your judgment solely on what the user would see in the assistant's message.
                - Ignore hidden or internal steps unless their results were explicitly communicated.

                2. **Goal completion**
                - The assistant must explicitly provide everything the user asked for.
                - If even one subpart of the task is missing, incomplete, or vague, the score must be **≤ 0.5**.

                3. **Correctness and relevance**
                - The information provided must be factually correct and directly relevant to the task.
                - Hallucinated or unrelated content automatically lowers the score.

                4. **Self-sufficiency**
                - The visible response must stand on its own; the user should not need prior context or follow-up clarification.

                5. **Strict bias toward failure**
                - When uncertain, assume the goal was **not achieved**.
                - The metric is designed to fail unless the assistant's output is precise, complete, and user-visible.

                SCORING GUIDE:

                - **1.0** → Goal completely and correctly achieved; all required outputs visible to the user.  
                - **0.75** → Mostly achieved; minor omissions or trivial inaccuracies.  
                - **0.5** → Partially achieved; core goal addressed, but key parts missing or incorrect.  
                - **0.25** → Weak attempt; loosely related but fails to satisfy the user’s request.  
                - **0.0** → Goal not achieved at all; irrelevant, wrong, or missing answer.

                *When in doubt, choose the lower score.*

                OUTPUT FORMAT:

                Return only a valid JSON object with this structure:

                {{
                    "score": 0.0,
                    "reason": "1-3 factual sentences explaining what parts of the user's goal were or were not achieved."
                }}

                The reason must:
                - Be objective and concise.
                - Refer to **specific missing or incorrect elements**.
                - Avoid vague language (“somewhat correct”, “pretty accurate”).

                EXAMPLES:

                **Example 1**
                Task: "Translate 'good night' into French."  
                Assistant Reply: "Bonne nuit."  
                →  
                {{
                    "score": 1.0,
                    "reason": "The assistant provided the exact, correct translation requested by the user."
                }}

                **Example 2**
                Task: "List three renewable energy sources."  
                Assistant Reply: "Solar and wind energy."  
                →  
                {{
                    "score": 0.5,
                    "reason": "The assistant only listed two sources instead of three, so the goal was partially achieved."
                }}

                **Example 3**
                Task: "Summarize this paragraph."  
                Assistant Reply: "It talks about technology."  
                →  
                {{
                    "score": 0.25,
                    "reason": "The summary is too vague and fails to convey key information from the text."
                }}

                *** END OF EXAMPLES ***

                USER TASK:
                {task}

                AGENT STEPS:
                {steps_taken}

                JSON:
            """
        )

    @staticmethod
    def get_plan_evaluation_score(task, steps_taken):
        return textwrap.dedent(
            f"""You are an expert evaluator assessing the **planning quality** and **plan adherence** of an AI agent tasked with fulfilling a user's request.

                OBJECTIVE:

                Evaluate:

                1. **Plan Quality** — Was the agent's plan clear, complete, and logically structured to fully address the user's task?  
                2. **Plan Adherence** — Did the agent consistently follow that plan without unjustified deviations, omissions, or extraneous steps?

                Your judgment must be strict: a plan must be well-formed and execution must align with it for a high score.

                EVALUATION CRITERIA

                - Plan Quality:  
                - The plan should explicitly or implicitly outline all necessary steps to fulfill the user's task.  
                - It must be logically ordered, neither vague nor overly generic.  
                - Missing critical components or unclear structuring lowers the score drastically.

                - Plan Adherence:  
                - Execution must closely match the planned steps.  
                - Any skipped, added, or rearranged steps without clear justification count as plan deviations.  
                - Minor, justified variations are acceptable but reduce the score slightly.

                - General Rules:  
                - If no discernible plan exists, score ≤ 0.5 regardless of task completion.  
                - Tool use should be coherent within the plan, not ad hoc or speculative.  
                - This evaluation excludes correctness or efficiency — focus solely on plan and adherence.

                SCORING GUIDE:

                - **1.0** → Complete, clear, and logical plan **fully followed** with all steps aligned to the user's goal.  
                - **0.75** → Mostly clear plan with minor omissions or small execution deviations that do not impact the overall strategy.  
                - **0.5** → Partial plan exists but is incomplete, vague, or only partially followed; notable deviations present.  
                - **0.25** → Weak or fragmented plan; execution frequently diverges or lacks coherence with any strategy.  
                - **0.0** → No evidence of a plan; execution appears random or unrelated to the user's task.

                INSTRUCTIONS:

                1. Identify the agent's plan from the steps taken (explicit plans stated or implicit structure).  
                2. Assess plan completeness and logical order relative to the user's task.  
                3. Compare execution steps against the plan to check for adherence, noting any unjustified deviations.  
                4. Deduct points for vagueness, missing critical steps, or inconsistent execution.

                OUTPUT FORMAT:

                Return only a valid JSON object with exactly two fields:

                {{
                    "score": 0.0,
                    "reason": "1-3 concise sentences explaining the quality of the plan and how well execution matched it. Specify missing or extra steps, plan clarity, and adherence issues."
                }}

                EXAMPLE:

                User Task: "Plan a business trip including booking a flight, hotel, and preparing an agenda."

                Agent Steps include:
                - Outlined flight, hotel, and agenda steps explicitly.
                - Executed flight and hotel booking steps.
                - Skipped agenda preparation despite mentioning it in the plan.

                Example JSON:

                {{
                    "score": 0.75,
                    "reason": "The agent formed a clear plan covering flights, hotel, and agenda, but failed to execute the agenda preparation step, reducing adherence."
                }}

                **** END OF EXAMPLE ****

                INPUTS:
                
                USER TASK:
                {task}

                AGENT STEPS:
                {steps_taken}

                JSON:
            """
        )

    @staticmethod
    def get_final_reason(
        final_score, threshold, goal_evaluations, plan_evalautions
    ):
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
