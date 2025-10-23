import textwrap
import json
from deepeval.tracing.utils import make_json_serializable


class PlanAdherenceTemplate:

    @staticmethod
    def extract_plan_from_trace(trace: dict) -> str:
        return textwrap.dedent(
            f"""You are a **systems analyst** evaluating an AI agent's execution trace.

                Your sole task is to extract the **explicit or clearly implied plan** the agent followed or intended to follow — *only if that plan is directly evidenced in the trace*.

                STRICT RULES TO FOLLOW:

                1. Source Evidence Requirement
                - Every plan step you include **must** be directly supported by explicit text in the trace.  
                - Acceptable evidence sources:
                    - `"reasoning"` or `"thought"` fields inside tool calls or function invocations.
                    - Explicit plan-like statements or lists written by the agent (e.g., “My plan is to…”).
                - If no evidence exists for a step, DO NOT infer or invent it.

                2. No Hallucination Policy 
                - You must *not* create or rephrase steps that aren't explicitly or strongly implied by the trace.  
                - If there is no coherent plan present, output an empty list.

                3. Focus on Intent, Not Outcomes
                - If the agent's plan is stated but execution differs, still extract the intended steps — but only if those intended steps are traceable.

                4. Granularity
                - Each step should represent a single distinct action or intention.
                - Avoid merging multiple intentions into one step or splitting one intention into multiple steps.

                5. Neutral Language
                - Reproduce the plan steps in **neutral, minimal paraphrasing**.  
                - Do not interpret motivation, quality, or success of actions.

                OUTPUT FORMAT:

                Return a JSON object with exactly this structure:
                {{
                    "plan": [
                        "step 1",
                        "step 2",
                        ...
                    ]
                }}

                If no plan is evidenced in the trace, return:
                {{
                    "plan": []
                }}

                Do not include commentary, confidence scores, or explanations.

                TRACE:

                {json.dumps(trace, indent=2, default=str)}

                JSON:
            """
        )

    @staticmethod
    def evaluate_adherence(
        user_task: str, agent_plan: str, execution_trace: dict
    ) -> str:
        return textwrap.dedent(
            f"""You are an **adversarial plan adherence evaluator**. Your goal is to assign the **lowest justifiable score** based on how strictly the agent's actions in the execution trace align with its declared plan.

                INPUTS:

                - **User Task:** The original request or objective.
                - **Agent Plan:** The explicit step-by-step plan the agent was supposed to follow.
                - **Execution Trace:** A detailed record of all agent actions, reasoning, tool calls, and outputs.

                EVALUATION OBJECTIVE:

                Determine whether the agent **exactly and exclusively** followed its plan.  
                You are not evaluating success, correctness, or usefulness — **only plan obedience**.

                Assume **non-adherence by default** unless clear, direct evidence in the trace proves that 
                each planned step was executed *as written* and *no additional actions occurred*.

                ### STRICT ADHERENCE RULES

                1. Step Verification
                - Every step in the plan must correspond to a **verifiable, explicit** action or reasoning entry in the trace.
                - Each step must appear in the same logical order as the plan.
                - If a step is missing, only implied, or ambiguous, treat as **not followed**.

                2. No Extraneous Actions
                - If the trace includes **any** major action, tool call, or reasoning segment not clearly present in the plan, immediately lower the score to as low as possible.
                - Extra or unnecessary steps are considered serious violations.

                3. Order Consistency
                - If the agent performed steps in a different order than the plan specifies, the score must be close to 0, regardless of other alignment.

                4. Completeness
                - If even one planned step is missing, skipped, or only partially reflected in the trace, the score must be lowest possible.

                5. Ambiguity Handling
                - If it is unclear whether a trace action corresponds to a plan step, treat that step as **not executed**.
                - When uncertain, assign the **lower score**.

                6. Focus Exclusively on Plan Compliance
                - Ignore task success, reasoning quality, or correctness of outcomes. 
                - Evaluate *only* whether the trace reflects the exact plan execution.


                SCORING SCALE

                - **1.0 — Perfect adherence**
                - Every planned step is explicitly and verifiably present in the trace, in correct order.
                - No skipped or added steps.
                - No ambiguity in matching.

                - **0.75 — Strong adherence**
                - All or nearly all steps are executed in order.
                - At most one minor deviation (e.g., a trivial reordering or minor redundant step) that does not change the plan’s structure.

                - **0.5 — Partial adherence**
                - Some steps clearly match, but others are missing, out of order, or replaced.
                - At least one extra or ambiguous action appears.
                - *This should be the highest score possible when there are any deviations.*

                - **0.25 — Weak adherence**
                - Only a few steps from the plan appear in the trace, or multiple extraneous actions occur.
                - The structure or sequence of the plan is mostly lost.

                - **0.0 — No adherence**
                - The trace shows little or no resemblance to the plan.
                - Steps are ignored, replaced, or executed in an entirely different order.

                Always err toward the **lower score** when evidence is partial, ambiguous, or contradictory.

                OUTPUT FORMAT:

                Return a JSON object with exactly this structure:

                {{
                    "score": 0.0,
                    "reason": "1-3 concise, factual sentences citing specific matched, missing, or extra steps."
                }}

                Requirements for `"reason"`:
                - Reference specific plan step numbers or phrases.
                - Mention concrete trace evidence of mismatches or additions.
                - Avoid subjective adjectives (e.g., “mostly”, “close”, “reasonable”).
                - Be precise and neutral.

                ---

                INPUTS:

                User Task:
                {user_task}

                Agent Plan:
                {agent_plan}

                Execution Trace:
                {json.dumps(execution_trace, indent=2, default=make_json_serializable)}

                ---

                JSON:
            """
        )
