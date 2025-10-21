import textwrap
import json
from deepeval.tracing.utils import make_json_serializable


class ExecutionEfficiencyTemplate:

    @staticmethod
    def extract_task_from_trace(trace: dict) -> str:
        return textwrap.dedent(
            f"""You are given a nested execution trace of a workflow involving an agent and its child components (e.g., tools, LLMs, retrievers, custom modules).

                Your task is to:
                1. Extract the underlying user task or objective, based only on the `input` field of the root-level agent.
                2. Based on the full trace, determine what specific sub-tasks or actions were performed by the agent and its children to fulfill that task.
                3. Return a fact-based, precise description of the user's goal, broken down into clear, concrete, and actionable components.

                Guidelines:
                - The extracted task should represent the user's intent, not the agent's internal reasoning or execution summary.
                - Use the trace itself (including tool/LLM/retriever/custom spans) to inform what sub-tasks were involved.
                - Be strictly objective. Do not use subjective language such as “successfully”, “efficiently”, or “appropriately”.
                - Describe the task from the user's point of view. Don't include internal agent behavior like “called flight_tool”.
                - Ensure the task description can be used as a reference to evaluate whether the agent's execution fulfilled the user's goal.

                CHAIN OF THOUGHT:
                1. Look at the root agent's `input` field — that contains the user's instruction or question.
                2. Review all children spans (`tool`, `llm`, `retriever`, `custom`, etc.) to see what the agent did.
                3. Infer the user's full task, combining the input with actions taken in the trace.
                4. Describe this task clearly and factually in plain language with exact actionable points.
                5. Do not speculate — only describe what is clearly supported by the trace.

                OUTPUT FORMAT:
                Return a JSON object with a single key:
                ```json
                {{
                "task": "..."
                }}
                ```
                If the task cannot be inferred, just return user's input
                ```json
                {{
                "task": user's input here taken from the trace
                }}
                ```

                EXAMPLE: 

                Trace:
                {{
                "name": "client_pitch_prep",
                "type": "agent",
                "input": {{
                    "input": "Help me prepare for a client pitch meeting in New York next Tuesday."
                }},
                "output": {{
                    "summary": "Pitch preparation started."
                }},
                "available_tools": ["flight_finder", "hotel_booker", "agenda_generator"],
                "children": [
                    {{
                    "name": "flight_finder",
                    "type": "tool",
                    "input": {{
                        "inputParameters": {{
                        "destination": "New York",
                        "date": "2024-08-13"
                        }}
                    }},
                    "output": {{
                        "flights": ["Flight A123", "Flight B456"]
                    }},
                    "children": []
                    }},
                    {{
                    "name": "hotel_booker",
                    "type": "tool",
                    "input": {{
                        "inputParameters": {{
                        "location": "New York",
                        "check_in": "2024-08-13",
                        "check_out": "2024-08-14"
                        }}
                    }},
                    "output": {{
                        "hotels": ["Central Hotel NY", "Eastside Suites"]
                    }},
                    "children": []
                    }},
                    {{
                    "name": "agenda_generator",
                    "type": "llm",
                    "input": {{
                        "prompt": "Draft a pitch agenda",
                        "input": [
                        {{
                            "role": "user",
                            "content": "Create an agenda for a client pitch meeting."
                        }}
                        ]
                    }},
                    "output": "1. Introduction\\n2. Product demo\\n3. Pricing discussion\\n4. Q&A",
                    "children": []
                    }},
                    {{
                    "name": "slide_fetcher",
                    "type": "retriever",
                    "input": {{
                        "embeddingInput": "pitch_deck_v2.pdf"
                    }},
                    "output": {{
                        "retrievalContext": ["Slide 2: Product Features", "Slide 5: Pricing Options"]
                    }},
                    "children": []
                    }}
                ]
                }}

                Example JSON:

                {{
                "task": "Prepare for a client pitch meeting in New York, including booking flights, reserving lodging, generating a presentation agenda, and retrieving relevant slides from a pitch deck."
                }}

                ===== END OF EXAMPLE ======

                **
                IMPORTANT: Please make sure to only return in JSON format with one key: 'task'
                **

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

                ---

                EXAMPLE:

                Task:
                Summarize the main argument in a paragraph of text provided by the user.

                Trace:
                The agent called an LLM to summarize the paragraph. No tools, no retrieval, no extra formatting.

                → JSON:
                {{
                "score": 1.0,
                "reason": "The agent used only a single LLM call to summarize the text, which directly fulfills the task with no unnecessary steps."
                }}

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
                - The agent must **ABSOLUTELY** not use a tool when it is 100% not required. Any extra tool calls or other activities that enhance an answer but slow the process should HEAVILY penalise the score.
                - It does not matter if the output of agent is correct, if the execution has extra steps that were not necessary, reduce the score

                ---

                TASK:
                {task}

                TRACE:
                {trace}

                JSON:
            """
        )
