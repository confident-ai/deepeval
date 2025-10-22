import textwrap
import json
from deepeval.tracing.utils import make_json_serializable


class PlanQualityTemplate:

    @staticmethod
    def extract_plan_from_trace(trace: dict) -> str:
        return textwrap.dedent(
            f"""You are a systems analyst tasked with extracting the **agent's internal plan** from its execution trace.

                You are given a full nested trace of an AI agent, including tools, LLM calls, retrievals, and custom components.

                Your job is to:
                1. Analyze the full execution trace to infer what the agent's intended **step-by-step plan** was for fulfilling the user's request.
                2. Return a structured natural language list of the plan the agent followed or intended to follow.
                3. Focus on what the agent seemed to **intend**, not just what it executed — for example, if some steps failed or were not completed.

                ---

                Output Format:
                Return a JSON object with a single key `"plan"` and a list of strings as the value, where each string is a step in the agent's intended plan.

                Example:
                {{
                    "plan": [
                        "Search for flights to Chicago on the requested date",
                        "Find hotels in Chicago for the duration of the trip",
                        "Generate a meeting agenda",
                        "Retrieve presentation slides from past decks"
                    ]
                }}

                Only include the plan. Do not add commentary, trace output, or evaluation.

                ---

                Trace:
                {json.dumps(trace, indent=2, default=str)}

                JSON:
            """
        )

    @staticmethod
    def evaluate_plan_quality(
        user_task: str, agent_plan: list, execution_trace: dict
    ) -> str:
        return textwrap.dedent(
            f"""You are an expert evaluator assessing the **quality of an agent's plan** to complete a user's task, using both the proposed plan and the full execution trace for context.

                You are given:
                - A structured description of the **user's task** — what the agent was supposed to accomplish.
                - The **agent's plan** — a sequence of steps intended to fulfill the user's task.
                - The full **execution trace** of the agent's behavior, which can provide additional context for understanding the scope and structure of the plan.

                Your job is to:
                1. Assess whether the agent's plan, **regardless of execution**, is a high-quality response to the user's task.
                2. Identify if the plan is:
                - **Complete**: Does it cover all major aspects of the user’s goal?
                - **Scoped**: Are the steps appropriately detailed, with no obvious gaps or unjustified complexity?
                - **Logical**: Are the steps ordered in a coherent and goal-directed manner?
                3. Use the trace **only as context** to help you understand the agent's operational environment — not to judge execution quality.

                ---

                SCORING GUIDE:

                - **1.0** → Excellent plan: Fully aligned with the task, no omissions, clean and purposeful structure.
                - **0.75** → Good plan: Mostly aligned with the task; may have minor inefficiencies or small gaps.
                - **0.5** → Mixed plan: Covers some of the task but has structural issues or partial coverage.
                - **0.25** → Poor plan: Many important steps missing or logic is flawed or unclear.
                - **0.0** → Inadequate plan: Plan is vague, unrelated, or fundamentally misaligned with the task.

                ---

                OUTPUT FORMAT:

                Return a JSON object like this:

                {{
                "score": 0.0,
                "reason": "..."  // 1-3 precise sentences explaining why the plan did or did not match the task.
                }}

                The reason must:
                - Be specific about what the plan did well or poorly.
                - Identify missing or unnecessary steps.
                - Avoid vague language like “looks good” or “reasonable”.

                ---

                USER TASK:
                {user_task}

                AGENT PLAN:
                {agent_plan}

                TRACE:
                {json.dumps(execution_trace, default=make_json_serializable, indent=2)}

                JSON:
            """
        )
