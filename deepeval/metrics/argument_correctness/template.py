from typing import List
from deepeval.test_case import ToolCall
import textwrap


class ArgumentCorrectnessTemplate:
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
                        "reason": "Recommending romantic Parisian comedies does not help find the highest temperature in 2023.",
                        "verdict": "no"
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
