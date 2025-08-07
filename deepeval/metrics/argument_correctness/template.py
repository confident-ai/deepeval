from typing import List
from deepeval.test_case import ToolCall, LLMTestCase
import textwrap


class ArgumentCorrectnessTemplate:
    @staticmethod
    def get_mcp_argument_correctness_prompt(
        test_case: LLMTestCase,
        available_primitives: str,
        primitives_used: str,
    ):
        return textwrap.dedent(
            f"""Evaluate whether the arguments passed to each tool (primitive) used by the agent were appropriate and correct for the intended purpose. Focus on whether the input types, formats, and contents match the expectations of the tools and are suitable given the user's request.

            You must return a JSON object with exactly two fields: 'score' and 'reason'.

            Scoring:
            - 'score' is a float between 0 and 1 inclusive.
            - Use intermediate values (e.g., 0.25, 0.5, 0.75) to reflect partial correctness, such as when argument types were correct but content was misaligned with intent.
            - 'reason' should clearly explain whether the arguments passed to tools were well-formed, appropriate, and aligned with the tool’s expected inputs and the user’s request.

            IMPORTANT:
            - Assume the selected tools themselves were appropriate (do NOT judge tool selection).
            - Focus ONLY on:
            - Whether the correct arguments were passed to each tool (e.g., types, structure, semantics).
            - Whether any required arguments were missing or malformed.
            - Whether extraneous, irrelevant, or incorrect values were included.
            - Refer to 'available_primitives' to understand expected argument formats and semantics.

            CHAIN OF THOUGHT:
            1. Understand the user’s request from 'test_case.input'.
            2. Review the arguments passed to each tool in 'primitives_used' (structure, content, type).
            3. Compare the arguments with what each tool in 'available_primitives' expects.
            4. Determine whether each tool was used with suitable and valid inputs, including values aligned with the task.
            5. Do NOT evaluate tool choice or output quality — only input correctness for the tools used.

            You must return only a valid JSON object. Do not include any explanation or text outside the JSON.

            -----------------
            User Input:
            {test_case.input}

            Agent Visible Output:
            {test_case.actual_output}

            Available Primitives (with expected arguments and signatures):
            {available_primitives}

            Primitives Used by Agent (with arguments passed):
            {primitives_used}

            Example Output:
            {{
                "score": 0.5,
                "reason": "The agent passed arguments of the correct type to all tools, but one tool received an input that did not match the user's intent and another had a missing required field."
            }}

            JSON:
            """
        )

    @staticmethod
    def generate_verdicts(input: str, tools_called: List[ToolCall]):

        stringified_tools_called = repr(tools_called)

        return textwrap.dedent(
            f"""
            For the provided list of tool calls, determine whether each tool call input parameter is relevantly and correctly addresses the input.

            Please generate a list of JSON with two keys: `verdict` and `reason`.
            The 'verdict' key should STRICTLY be either a 'yes' or 'no'. Answer 'yes' if the tool call input parameter is relevantly and correctly addresses the original input, 'no' if the tool call input parameter doesn't correctly and relevantly address the original input.
            The 'reason' is the reason for the verdict.
            Provide a 'reason' ONLY if the answer is 'no'. 
            If there is no input parameter, answer 'no' for the verdict and provide the reason as "No input parameter provided".

            **
            IMPORTANT: Please make sure to only return in valid and parseable JSON format, with the 'verdicts' key mapping to a list of JSON objects. Ensure all strings are closed appropriately. Repair any invalid JSON before you output it.
            Example input: 
            "What was the highest temperature recorded in Paris in 2023?"
            
            Example tool calls: 
            [
                ToolCall(
                    name="WeatherHistoryAPI",
                    description="Fetches historical weather data for a given city and date range",
                    reasoning="I need to check all 2023 temperature records for Paris to find the highest one.",
                    input_parameters={{
                        "city_name": "Paris",
                        "country_code": "FR",
                        "date_range_start": "2023-01-01",
                        "date_range_end": "2023-12-31",
                        "data_type": "temperature_max_daily_celsius"
                    }}
                ),
                ToolCall(
                    name="MathAnalyzer",
                    description="Performs statistical calculations on numeric datasets",
                    reasoning="I will calculate the maximum temperature value from the daily dataset.",
                    input_parameters={{
                        "operation": "max",
                        "dataset_source": "WeatherHistoryAPI.daily_max_temperatures",
                        "expected_unit": "celsius"
                    }}
                ),
                ToolCall(
                    name="MovieRecommender",
                    description="Recommends movies based on user mood or location",
                    reasoning="I thought Paris movies might be fun to suggest, but this is unrelated to the question.",
                    input_parameters={{
                        "preferred_genres": ["romance", "comedy"],
                        "setting_city": "Paris",
                        "language_preference": "French or English"
                    }}
                )
            ]

            Example JSON:
            {{
                "verdicts": [
                    {{
                        "verdict": "yes"
                    }},
                    {{
                        "verdict": "yes"
                    }},
                    {{
                        "verdict": "no",
                        "reason": "Recommending romantic Parisian comedies does not help find the highest temperature in 2023."
                    }}
                ]  
            }}
            ===== END OF EXAMPLE ======

            Since you are going to generate a verdict for each statement, the number of 'verdicts' SHOULD BE STRICTLY EQUAL to the number of `statements`.
            **          

            Input:
            {input}

            Tool Calls:
            {stringified_tools_called}

            JSON:
            """
        )

    @staticmethod
    def generate_reason(
        incorrect_tool_calls_reasons: List[str], input: str, score: float
    ):
        return textwrap.dedent(
            f"""Given the argument correctness score, the list of reasons of incorrect tool calls, and the input, provide a CONCISE reason for the score. Explain why it is not higher, but also why it is at its current score. You can mention tool calls or input, but do not mention an output or a response.
            If there is nothing incorrect, just say something positive with an upbeat encouraging tone (but don't overdo it otherwise it gets annoying).

            **
            IMPORTANT: Please make sure to only return in JSON format, with the 'reason' key providing the reason. Ensure all strings are closed appropriately. Repair any invalid JSON before you output it.

            Example:
            Example JSON:
            {{
                "reason": "The score is <argument_correctness_score> because <your_reason>."
            }}
            ===== END OF EXAMPLE ======
            **


            Argument Correctness Score:
            {score}

            Reasons why the score can't be higher based on incorrect tool calls:
            {incorrect_tool_calls_reasons}

            Input:
            {input}

            JSON:
             """
        )
