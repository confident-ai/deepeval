import textwrap
import json
from deepeval.tracing.utils import make_json_serializable


class PlanAdherenceTemplate:

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
    def evaluate_adherence(
        user_task: str, agent_plan: str, execution_trace: dict
    ) -> str:
        return textwrap.dedent(
            f"""You are an expert evaluator assessing the **plan adherence** of an AI agent system.

                You are given:
                - A structured description of the **user's original task**.
                - A **step-by-step plan** that the agent was expected to follow to complete that task.
                - A full **execution trace**, showing all tools, LLMs, retrievers, and other components used by the agent.

                Your job is to:
                1. Compare the expected plan against the actual actions taken by the agent in the trace.
                2. Determine to what extent the agent **followed the intended plan** in structure, sequence, and completeness.
                3. Identify:
                - Steps from the plan that were fully followed.
                - Steps that were skipped, altered, or reordered.
                - Any additional actions not present in the plan.

                ---

                SCORING GUIDE:

                - **1.0** → Perfect adherence: Every step in the plan was executed as intended, in logical order, with no extra or missing actions.
                - **0.75** → Strong adherence: Most steps were followed, possibly with minor reordering or minor unnecessary additions.
                - **0.5** → Partial adherence: Several steps were skipped or altered, or multiple unplanned actions were added.
                - **0.25** → Weak adherence: Only a few planned steps were followed; execution diverged significantly from the original plan.
                - **0.0** → No adherence: Agent actions do not resemble the original plan in structure or content.

                ---

                OUTPUT FORMAT:

                Return a **JSON object** like this:

                {{
                    "score": 0.0,
                    "reason": "..."  // 1-3 sentences explaining how closely the execution matched the plan.
                }}

                The reason must:
                - Point to specific matches or deviations from the plan.
                - Reference missed, reordered, or added steps.
                - Avoid subjective terms like “generally good” or “fairly close”.

                ---

                USER TASK:
                {user_task}

                AGENT PLAN:
                {agent_plan}

                EXECUTION TRACE:
                {json.dumps(execution_trace, default=make_json_serializable, indent=2)}

                JSON:
            """
        )
