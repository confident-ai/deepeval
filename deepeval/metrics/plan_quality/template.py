import textwrap
import json
from deepeval.tracing.utils import make_json_serializable


class PlanQualityTemplate:

    @staticmethod
    def evaluate_plan_quality(user_task: str, agent_plan: list) -> str:
        return textwrap.dedent(
            f"""You are a **plan quality evaluator**. Your task is to critically assess the **quality, completeness, and optimality** of an AI agent's plan to accomplish the given user task.

                INPUTS:

                - **User Task:** The user's explicit goal or instruction.
                - **Agent Plan:** The ordered list of steps the agent intends to follow to achieve that goal.

                EVALUATION OBJECTIVE:

                Judge the **intrinsic quality** of the plan — whether the plan itself is strong enough to fully and efficiently achieve the user's task.

                The evaluation must be **strict**.  If the plan is incomplete, inefficient, redundant, or missing critical details, assign a very low score.

                STRICT EVALUATION CRITERIA:

                1. Completeness (Most Important)
                - The plan must fully address all major requirements of the user task.  
                - Missing even one critical subtask or dependency should reduce the score sharply.
                - The plan must include all prerequisite actions necessary for the final outcome.

                2. Logical Coherence
                - Steps must follow a clear, rational sequence that leads directly to completing the task.  
                - Disordered, redundant, or circular reasoning should be penalized heavily.
                - Every step must have a clear purpose; no filler or irrelevant actions.

                3. Optimality and Efficiency
                - The plan must be **minimal but sufficient** — no unnecessary or repetitive steps.
                - If a more direct, simpler, or logically superior plan could achieve the same outcome, the current plan should receive a lower score.

                4. Level of Detail
                - Each step should be specific enough for an agent to execute it reliably without ambiguity.  
                - Vague steps (e.g., “Do research”, “Handle results”) that lack operational clarity 
                    lower the score.

                5. Alignment with Task
                - The plan must explicitly and directly target the user's stated goal.  
                - If any step diverges from the main objective, the score should drop significantly.

                ---

                SCORING SCALE (STRICT)

                - **1.0 — Excellent plan**
                - Fully complete, logically ordered, and optimally efficient.  
                - No missing, redundant, or ambiguous steps.  
                - Directly fulfills every aspect of the user task.

                - **0.75 — Good plan**
                - Covers nearly all aspects of the task with clear logic.  
                - Minor gaps or small inefficiencies that do not block task completion.

                - **0.5 — Adequate but flawed plan**
                - Partially complete; key details missing or step order inefficient.  
                - Some ambiguity or redundancy that would likely affect execution success.

                - **0.25 — Weak plan**
                - Major missing steps or unclear logic.  
                - The plan would likely fail to complete the task as written.

                - **0.0 — Inadequate plan**
                - Irrelevant, incoherent, or severely incomplete plan.  
                - Does not align with the user’s task or cannot plausibly achieve it.

                *When in doubt, assign the lower score.*

                OUTPUT FORMAT:

                Return a JSON object with this exact structure:

                {{
                    "score": 0.0,
                    "reason": "1-3 short, precise sentences explaining what the plan lacks or how it could fail."
                }}

                The `"reason"` must:
                - Reference specific missing, unclear, or inefficient steps.
                - Avoid vague language (“seems fine”, “mostly works”).
                - Use objective terms describing gaps or weaknesses.

                PROVIDED DATA

                User Task:
                {user_task}

                Agent Plan:
                {agent_plan}


                JSON:
            """
        )
