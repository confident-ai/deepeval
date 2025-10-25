import textwrap
import json
from deepeval.tracing.utils import make_json_serializable


class StepEfficiencyTemplate:

    @staticmethod
    def extract_task_from_trace(trace: dict) -> str:
        return textwrap.dedent(
            f"""You are a **trace analyst** tasked with extracting the **user's original goal or task** from a complete nested execution trace of an AI agent.

                YOUR OBJECTIVE:

                Identify and describe **exactly what the user asked the agent to do**, based only on the user's explicit input and any unambiguous contextual details present in the trace.

                Your goal is to produce a **concise, fact-based statement** that captures the *intended user task* — not the agent's plan, actions, reasoning, or assumptions.

                STRICT EXTRACTION RULES:

                1. Primary Source: Root-Level User Input
                - The user's task must be derived **directly and primarily** from the root agent's `"input"` field.  
                - If that field contains nested `"input"` or `"messages"`, extract the true user instruction or request text from within it.

                2. Secondary Context: Subtasks as Clarifiers (Optional)
                - You may use child spans (tools, retrievers, LLMs) **only** to clarify or disambiguate what the user explicitly asked for — 
                    e.g., to confirm that the task involves multiple subtasks the user clearly implied (like booking and planning steps for a trip).
                - You may **NOT** infer new goals that the user did not state or imply.

                3. No Hallucination
                - Do **NOT** invent goals, assumptions, or implied needs beyond what is explicitly or clearly inferable from the input.
                - If the user's request is vague, preserve that vagueness — do not expand it.

                4. Agent-Agnostic Rule
                - Ignore the agent's tools, methods, reasoning, or internal operations.
                - The task reflects **what the user wanted**, not how the agent chose to approach it.

                5. Perspective
                - Express the extracted task **from the user's perspective**, as if restating what they asked the system to do.
                - Avoid any meta or evaluative phrasing (“The user wanted the agent to…”).

                6. Fallback Condition
                - If the only available information about the task is the raw user input text, return that input verbatim without modification.

                OUTPUT FORMAT:

                Return **only** a JSON object of this form:

                {{
                    "task": "<a single clear sentence summarizing the user's explicit goal>"
                }}

                - The `"task"` value should be a single, coherent natural language sentence or two at most.
                - Do not include commentary, metadata, or any additional fields.

                EXAMPLES:

                Example Trace: {{ 
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
                                }} }}, 
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
                                }} }}, 
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

                IMPORTANT ENFORCEMENT RULES:

                - If multiple user inputs exist, identify the overall task that user has in mind.
                - Do not include execution details, tools, function names, or reasoning text.
                - Avoid restating or paraphrasing beyond clarity; preserve the user's intent exactly.
                - When uncertain, extract **less rather than more** — prefer minimal, factual phrasing over speculative completion.

                TRACE DATA:

                {json.dumps(trace, default=make_json_serializable, indent=2)}

                ---

                ### JSON:
            """
        )

    @staticmethod
    def get_execution_efficiency(task: str, trace: dict) -> str:
        return textwrap.dedent(
            f"""You are an **efficiency auditor** evaluating how economically an AI agent executed a task.

                OBJECTIVE:

                Determine how **efficiently** the agent executed the given task based on its full execution trace.
                Efficiency means achieving the user's goal using the **fewest, simplest, and most direct** actions possible.

                You must assign a score from **0.0 to 1.0** that reflects how close the execution came to the *minimal necessary sequence of actions*.

                **Important:** You are not evaluating correctness, completeness, creativity, or helpfulness — only the *efficiency* of the execution.

                STRICT EVALUATION RULES:

                1. Zero-Tolerance for Unnecessary Actions
                - Every step, tool call, LLM query, or retrieval must be **strictly required** to fulfill the task.  
                - If a single tool, retrieval, or reasoning step is superfluous, speculative, repetitive, or stylistic, 
                    the score must be as low as possible, regardless of outcome quality.
                - Adding “helpful” or “contextual” actions that were not explicitly necessary is an inefficiency.

                2. Minimal Action Principle
                - The ideal execution performs the **exact minimum number of steps** needed to complete the task.  
                - Each step must directly contribute to completing the task, not to exploration, confirmation, or elaboration.

                3. No Speculation or Enrichment
                - Any activity aimed at *enhancing*, *expanding*, or *beautifying* the answer (e.g., extra retrievals, style edits, rephrasings) 
                    reduces the score sharply (≤ 0.25).  
                - Efficiency is about restraint — **doing exactly what's required, nothing more**.

                4. Directness and Focus
                - Steps must appear in a logically minimal sequence from input to goal.  
                - Repetition, re-querying, nested reasoning loops, or tool reuse when not needed 
                    indicate inefficiency.

                5. Resource Economy
                - Use of multiple LLM calls, retrievers, or tools when one would suffice must be penalized.
                - Avoided resources (if the agent achieved the task through simpler means) improve efficiency.

                6. When in Doubt
                - If it is unclear whether an action was required or not, **assume it was unnecessary** and lower the score.
                - Err on the side of penalizing over generosity.

                SCORING SCALE (STRICT)

                - **1.0 — Perfectly efficient**
                - Only essential steps taken.  
                - Each action was directly necessary for task completion.  
                - No speculative, redundant, or decorative work.

                - **0.75 — Strong efficiency**
                - Mostly minimal execution with one small redundant or stylistic step.  
                - Slight overuse of a tool or repeated call, but otherwise tight.

                - **0.5 — Moderate efficiency**
                - Noticeable inefficiency: extra steps, unnecessary tool calls, or indirect methods.  
                - The same task could clearly have been completed faster or with fewer actions.

                - **0.25 — Low efficiency**
                - Multiple irrelevant or unjustified actions taken.  
                - Execution path significantly longer or more complex than needed.

                - **0.0 — Highly inefficient**
                - Execution was verbose, exploratory, speculative, or wasteful.  
                - Most actions were unnecessary or unrelated to achieving the core task.

                *When uncertain, always assign the lower score.*

                OUTPUT FORMAT:

                Return a single JSON object in this exact format:

                {{
                    "score": 0.0,
                    "reason": "1-3 concise factual sentences describing where inefficiencies occurred."
                }}

                The `reason` must:
                - Identify specific inefficient actions (e.g., redundant LLM call, unnecessary retrieval, speculative tool use).
                - Avoid subjective phrasing (“reasonable”, “seems okay”, “somewhat efficient”).
                - Be direct and concrete: “Extra retrieval used for enrichment”, “Multiple summarizations of same data”, etc.

                EXAMPLES

                **Example 1:**
                Task: "Summarize the given text."
                Trace: Agent calls an LLM twice, then performs an extra web search.

                → Output:
                {{
                    "score": 0.25,
                    "reason": "The agent used redundant LLM calls and performed an unnecessary web search. Only one LLM call was required for the summary."
                }}

                **Example 2:**
                Task: "Convert a date to ISO format."
                Trace: Agent performs one computation directly.

                → Output:
                {{
                    "score": 1.0,
                    "reason": "The agent completed the task with one minimal action and no unnecessary steps."
                }}

                FINAL REMINDERS

                - Efficiency = minimality. Any extra work, enrichment, or indirect approach must lower the score.
                - Do not consider correctness, helpfulness, or reasoning quality.
                - A “good answer” can still score **0.0** if it was achieved inefficiently.
                - This metric is adversarial: assign the lowest score possible unless execution was provably minimal.

                TASK:
                {task}

                TRACE:
                {json.dumps(trace, indent=2, default=str)}

                JSON:
            """
        )
