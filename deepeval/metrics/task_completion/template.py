from deepeval.metrics.utils import print_tools_called
import textwrap
import json
from deepeval.tracing.utils import make_json_serializable


class TaskCompletionTemplate:

    # TODO: Deprecate this function soon
    @staticmethod
    def extract_goal_and_outcome(
        input: str, actual_output: str, tools_called: list
    ) -> str:
        return textwrap.dedent(
            f"""Given an agentic workflow comprised of a human input, AI response, and tools used by the AI, identify the task (or objective the user wants to achieve) and the task_outcome (the final outcome or result of the workflow).
            The task outcome should be solely factual, derived strictly from the workflow (input, response, and tools called), without any reasoning involved.

            ``Example:
            Example input: Can you help me plan a trip to New York this weekend, including travel, accommodation, and sightseeing?
            Example tools called:
            [
                {{
                    "name": "flight_search",
                    "description": "Search for flights based on destination and date.",
                    "reasoning": "The input specifies travel as part of the task. This tool is needed to find flight options based on the user's destination and dates.",
                    "output": {{
                        "flights": ["Flight A", "Flight B"]
                    }},
                    "input_parameters": {{
                        "destination": "New York",
                        "date": "Saturday",
                        "return_date": "Sunday"
                    }}
                }},
                {{
                    "name": "hotel_search",
                    "description": "Search for hotels in the given location.",
                    "reasoning": "The input specifies accommodation as part of the task. This tool is needed to find hotel options in the specified location for the provided dates.",
                    "output": {{
                        "hotels": ["Grand NY Hotel", "Empire Suites"]
                    }},
                    "input_parameters": {{
                        "location": "New York",
                        "check_in": "Saturday",
                        "check_out": "Sunday"
                    }}
                }},
                {{
                    "name": "sightseeing_search",
                    "description": "Provide sightseeing options for a given location.",
                    "reasoning": "The input specifies sightseeing as part of the task. This tool is needed to generate a list of recommended places to visit in New York.",
                    "output": {{
                        "sights": ["Central Park", "Statue of Liberty", "Times Square"]
                    }},
                    "input_parameters": {{
                        "location": "New York"
                    }}
                }}
            ]
            Example response: Sure! Flights available to New York include Flight A and Flight B. Accommodation options include Grand NY Hotel and Empire Suites. Suggested sightseeing spots in New York are Central Park, Statue of Liberty, and Times Square. 

            Example JSON:
            {{
                "task": "Have the system plan a weekend trip to New York, including travel, accommodation, and sightseeing.",
                "outcome": "The system provided suggested flights departing on Saturday and returning on Sunday, identified hotels with check-in on Saturday and check-out on Sunday, and generated a list of sightseeing destinations in New York City."
            }}
            ===== END OF EXAMPLE ======
                    
            **
            IMPORTANT: Please make sure to only return in JSON format with two keys: `task` and `outcome`.
            **

            input: {input}
            tools called:
            {print_tools_called(tools_called)}
            response: {actual_output}

            JSON:
        """
        )

    @staticmethod
    def extract_task_and_outcome_from_trace(trace: dict) -> str:
        return textwrap.dedent(
            f"""Given a nested workflow trace whose spans may be of type `agent`, `tool`, `llm`, `retriever`, or `custom`, identify:

            1. **task** – the task or objective expressed by the user in the root agent’s input.  
            2. **outcome** – a strictly factual description of what the system did, based only on the trace.

            The task outcome should be solely factual, derived strictly from the trace.
            Do **not** include subjective language such as “successfully”, “efficiently”, or “well”.  
            Enumerate each relevant action or output the trace shows, in plain language.

            ``Example:
            Example trace:
            {{
            "name": "trip_planner",
            "type": "agent",
            "input": {{
                "input": "Can you help me plan a business trip to Chicago next week?"
            }},
            "output": {{
                "summary": "Trip planning initiated."
            }},
            "available_tools": ["flight_tool", "hotel_tool"],
            "agent_handoffs": [],
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
                "description": "Search for flights to a destination",
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
                "description": "Find hotels for specified dates",
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
                "model": "gpt-4",
                "inputTokenCount": 38,
                "outputTokenCount": 21,
                "children": []
                }},
                {{
                "name": "slide_retriever",
                "type": "retriever",
                "input": {{
                    "embeddingInput": "presentation.pptx"
                }},
                "output": {{
                    "retrievalContext": ["Slide 1: Revenue", "Slide 2: Client Feedback"]
                }},
                "topK": 3,
                "chunkSize": 512,
                "children": []
                }},
                {{
                "name": "client_embedder",
                "type": "custom",
                "input": {{
                    "text": "Concerns from enterprise clients"
                }},
                "output": [0.1, 0.32, 0.85],
                "children": []
                }}
            ]
            }}
            Example JSON:
            {{
            "task": "Plan a business trip to Chicago, including flights, lodging, meeting agenda, presentation review, and client preparation.",
            "outcome": "The system invoked a tool to retrieve two flight options and another tool to find two hotels for the specified dates. An LLM with model 'gpt-4' generated a three-topic meeting agenda from a system/user prompt. A retriever extracted three slides using the embedding input 'presentation.pptx' with topK=3 and chunk size 512. A custom component generated vector embeddings for a client-related input string."
            }}
            ===== END OF EXAMPLE =====

            **
            IMPORTANT – return only valid JSON with two keys: `task` and `outcome`.
            **

            trace:
            {json.dumps(trace, default=make_json_serializable, indent=2)}

            JSON:
            """
        )

    @staticmethod
    def generate_verdict(task: str, actual_outcome: str):
        return textwrap.dedent(
            f"""Given the task (desired outcome) and the actual achieved outcome, compare how well the actual outcome aligns with the desired task.

                Please return a JSON with two keys: `verdict` and `reason`.
                - The `verdict` should be a score from 0 to 1, where 1 indicates the actual outcome perfectly achieves the desired task, and 0 indicates it does not achieve the task at all.
                - The `reason` should explain why the given verdict was assigned.

                **
                IMPORTANT: Please make sure to only return in JSON format, with `verdict` as a float between 0 and 1.
                Example:
                Task: Have the system plan a weekend trip to New York, including travel, accommodation, and sightseeing.
                Actual outcome: The system provided suggested flights departing on Saturday and returning on Sunday, identified hotels with check-in on Saturday and check-out on Sunday, and generated a list of sightseeing destinations in New York City.
                Example JSON:
                {{
                    "verdict": 0.85,
                    "reason": "The system suggested flights, accommodation, and sightseeing options but did not fully plan the trip as expected."
                }}
                **

                Task:
                {task}

                Actual outcome:
                {actual_outcome}

                JSON:
            """
        )
