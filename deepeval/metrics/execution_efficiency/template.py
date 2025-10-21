import textwrap
import json
from deepeval.tracing.utils import make_json_serializable


class ExecutionEfficiencyTemplate:

    @staticmethod
    def extract_task_from_trace(trace: dict) -> str:
        return textwrap.dedent(
            f"""You are an expert evaluator tasked with extracting the **user's original goal** from a nested execution trace of an AI agent's workflow.

                You are given:
                - A **trace**: a full nested dictionary representing the entire execution of an agent and its child components (tools, LLM calls, retrievers, custom modules, etc.).
                - The trace contains a root-level agent with an `input` field representing the user's original instruction or query.

                Your job is to:
                1. Identify the user's **full intended task or objective** based strictly on the root-level agent's `input` field.
                2. Analyze all child spans in the trace to understand what sub-tasks or actions were performed.
                3. Return a **precise, fact-based, structured description** of the user's goal, expressed from the user's perspective as a clear set of concrete, actionable points.
                4. Do **not** include any internal agent reasoning, intermediate steps, or component names.
                5. Do **not** speculate beyond what the trace clearly supports.
                6. If the user's task cannot be inferred beyond the root input, return that input verbatim.

                ---

                GUIDELINES:

                - The extracted task should reflect only the user's intent.
                - Use the full trace only to inform what sub-tasks or subtleties the user requested.
                - Avoid subjective or evaluative language.
                - Produce output that can serve as a definitive reference to check whether the agent fulfilled the user's request.

                ---

                OUTPUT FORMAT:

                Return a JSON object with a single key `"task"` and a string value describing the user's task.

                Use this exact format with double curly braces for the example:

                ---

                Example Trace:

                {{
                "name": "trip_planner",
                "type": "agent",
                "input": {{
                    "input": "Help me plan a business trip to Chicago next week."
                }},
                "children": [
                    {{
                    "name": "flight_tool",
                    "type": "tool",
                    "input": {{
                        "inputParameters": {{
                        "destination": "Chicago",
                        "date": "2024-07-10"
                        }}
                    }},
                    "output": {{
                        "flights": ["Flight 101", "Flight 202"]
                    }},
                    "children": []
                    }},
                    {{
                    "name": "hotel_tool",
                    "type": "tool",
                    "input": {{
                        "inputParameters": {{
                        "location": "Chicago",
                        "check_in": "2024-07-10",
                        "check_out": "2024-07-12"
                        }}
                    }},
                    "output": {{
                        "hotels": ["The Grand Chicago", "Lakeview Inn"]
                    }},
                    "children": []
                    }},
                    {{
                    "name": "agenda_llm",
                    "type": "llm",
                    "input": {{
                        "prompt": "Draft a meeting agenda",
                        "input": [
                        {{
                            "role": "system",
                            "content": "You are an executive assistant."
                        }},
                        {{
                            "role": "user",
                            "content": "Create an agenda for a client strategy meeting."
                        }}
                        ]
                    }},
                    "output": "1. Q2 review\\n2. Client feedback\\n3. Strategy planning",
                    "children": []
                    }}
                ]
                }}

                Expected JSON:

                {{
                    "task": "Plan a business trip to Chicago next week, including booking a flight, reserving a hotel, and drafting a client meeting agenda."
                }}

                ---

                **IMPORTANT**: Only return a single JSON object with the `"task"` key and your extraction as the value.

                Trace:

                {json.dumps(trace, default=make_json_serializable, indent=2)}

                JSON:
            """
        )

    def get_execution_efficiency(task: str, trace: dict) -> str:
        return textwrap.dedent(
            f"""You are an expert evaluator assessing the **execution efficiency** of an AI agent.

                You are given:
                - A **task**: a structured, factual description of what the user intended to accomplish.
                - A **trace**: a full execution trace showing all components (tools, LLM calls, retrievals, custom functions, etc.) invoked by the agent to fulfill that task.

                Your job is to assign an **efficiency score** from 0.0 to 1.0 based on how precisely and minimally the agent completed the task — **regardless of final output quality**.

                ---

                DEFINITION: Execution efficiency refers to how economically the agent used available resources to accomplish the given task.

                The most efficient executions:
                - Use **only** components that are **strictly required** to fulfill the task.
                - **Avoid** any extra, speculative, redundant, or stylistic steps.
                - Follow a **logical, minimal, and goal-directed** sequence of actions.
                - Do **not** attempt to “enrich” the response with unrelated insights, external lookups, or commentary unless explicitly required by the task.
                - Demonstrate **surgical minimalism**: each action must be **justified by necessity**, not helpfulness.

                ---

                INSTRUCTIONS:

                Step 1: Read the **task** carefully. Identify what information or deliverables the user asked for — and only that.

                Step 2: Analyze the **trace**:
                - What tools or components were used?
                - Were they required to fulfill the task, or were they optional or speculative?
                - Could the same output have been achieved **without** certain steps?
                - Were there **missing** steps that should have been included?

                Step 3: Identify violations of efficiency:
                - **Overuse**: Any step that was not strictly required — even if it improved the output — must reduce the score.
                - **Misuse**: Use of a tool/LLM/retriever for something that could’ve been done more directly.
                - **Underuse**: Omitted steps that were necessary to fulfill the task.

                ---

                SCORING GUIDE:

                - **1.0** → Perfectly efficient: All steps were strictly required. No redundancy, no speculation, no omissions.
                - **0.75** → Mostly efficient: Minor inefficiency (e.g. a redundant LLM step or unnecessary formatting), but task execution was mostly tight.
                - **0.5** → Mixed: Some steps were useful but unnecessary; task could have been completed more directly.
                - **0.25** → Low efficiency: Several irrelevant or unjustified components were used. Core task was overcomplicated or indirect.
                - **0.0** → Inefficient: Execution was verbose, indirect, exploratory, or speculative; many unnecessary steps or missing essentials.

                ---

                OUTPUT FORMAT:

                Return a **JSON object** like this:

                {{
                "score": 0.0,
                "reason": "..."  // 1-3 precise sentences explaining your score.
                }}

                The reason must:
                - Be specific about which actions were unnecessary or inefficient.
                - Avoid vague language like "pretty good" or "reasonable".
                - Justify deductions clearly: e.g. “LLM was used to restate input”, or “web search added speculative context”.

                EXAMPLE:

                Task:
                Answer a direct question using only the information explicitly provided by the user.

                Trace:
                The agent parsed the answer from the input, but then invoked external tools to supplement the response with speculative or enriching content.

                → JSON:
                {{  
                    "score": 0.25,  
                    "reason": "The agent completed the task using unnecessary external tools. The output included additional context not required by the task, which reduced execution efficiency."  
                }}

                ---

                **IMPORTANT**:
                - Do not consider output quality, correctness, or helpfulness — this metric only evaluates execution efficiency.
                - Do not invent or infer intent beyond what is clearly stated in the task.
                - The agent **MUST ABSOLUTELY NOT** use a tool UNLESS it is 100% required to finsih the task given. Any extra tool calls or other activities that enhance an answer but slow the process should HEAVILY penalise the score.
                - This metric is built to fail, ONLY give a high score if the agent has done everyting **meticulously** right. If not, give the lowest score possible

                ---

                TASK:
                {task}

                TRACE:
                {trace}

                JSON:
            """
        )
