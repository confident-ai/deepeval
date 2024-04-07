import pytest
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval import assert_test
from langchain.schema import AIMessage
from unittest import mock
import os


@pytest.mark.skip(reason="openai is expensive")
def test_g_eval():
    metric = GEval(
        name="Validity",
        criteria="The response is a valid response to the prompt",
        threshold=0.6,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
    )
    test_case = LLMTestCase(
        input="What is the capital of France?",
        actual_output="Paris",
        expected_output="Paris",
        context=["Geography"],
    )
    assert_test(test_case, [metric])


@pytest.mark.skip(reason="openai is expensive")
@mock.patch.dict(
    os.environ,
    {
        "OPENAI_API_KEY": "test",
    },
)
def test_generate_logprobs_based_score():
    metric = GEval(
        name="Validity",
        criteria="The response is a valid response to the prompt.",
        threshold=0.5,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
    )
    raw_response = AIMessage(
        content='```json\n{\n  "score": 9,\n  "reason": "Omitted"\n}\n```',
        response_metadata={
            "finish_reason": "stop",
            "logprobs": {
                "content": [
                    {
                        "token": "```",
                        "bytes": [96, 96, 96],
                        "logprob": -0.018150061,
                        "top_logprobs": [
                            {
                                "token": "```",
                                "bytes": [96, 96, 96],
                                "logprob": -0.018150061,
                            },
                            {
                                "token": "{\n",
                                "bytes": [123, 10],
                                "logprob": -4.01815,
                            },
                            {
                                "token": "``",
                                "bytes": [96, 96],
                                "logprob": -16.3619,
                            },
                            {
                                "token": '{"',
                                "bytes": [123, 34],
                                "logprob": -18.64315,
                            },
                            {
                                "token": " ```",
                                "bytes": [32, 96, 96, 96],
                                "logprob": -23.2994,
                            },
                            {"token": "{", "bytes": [123], "logprob": -24.7369},
                            {"token": "\n", "bytes": [10], "logprob": -24.7369},
                            {
                                "token": ">{\n",
                                "bytes": [62, 123, 10],
                                "logprob": -25.9869,
                            },
                            {"token": "`", "bytes": [96], "logprob": -26.4869},
                            {
                                "token": "{\n\n",
                                "bytes": [123, 10, 10],
                                "logprob": -26.7369,
                            },
                            {
                                "token": "   ",
                                "bytes": [32, 32, 32],
                                "logprob": -28.01815,
                            },
                            {
                                "token": " {\n",
                                "bytes": [32, 123, 10],
                                "logprob": -28.14315,
                            },
                            {
                                "token": "json",
                                "bytes": [106, 115, 111, 110],
                                "logprob": -29.2369,
                            },
                            {
                                "token": "**",
                                "bytes": [42, 42],
                                "logprob": -29.83065,
                            },
                            {
                                "token": "\ufeff",
                                "bytes": [239, 187, 191],
                                "logprob": -30.158775,
                            },
                            {
                                "token": "{}{\n",
                                "bytes": [123, 125, 123, 10],
                                "logprob": -30.752525,
                            },
                            {
                                "token": "{\r\n",
                                "bytes": [123, 13, 10],
                                "logprob": -30.95565,
                            },
                            {
                                "token": "}{\n",
                                "bytes": [125, 123, 10],
                                "logprob": -31.221275,
                            },
                            {
                                "token": "//{\n",
                                "bytes": [47, 47, 123, 10],
                                "logprob": -31.471275,
                            },
                            {
                                "token": "{\n\n\n",
                                "bytes": [123, 10, 10, 10],
                                "logprob": -31.70565,
                            },
                        ],
                    },
                    {
                        "token": "json",
                        "bytes": [106, 115, 111, 110],
                        "logprob": 0.0,
                        "top_logprobs": [
                            {
                                "token": "json",
                                "bytes": [106, 115, 111, 110],
                                "logprob": 0.0,
                            },
                            {
                                "token": "JSON",
                                "bytes": [74, 83, 79, 78],
                                "logprob": -17.65625,
                            },
                            {
                                "token": " json",
                                "bytes": [32, 106, 115, 111, 110],
                                "logprob": -22.03125,
                            },
                            {
                                "token": " \n",
                                "bytes": [32, 10],
                                "logprob": -23.625,
                            },
                            {
                                "token": "{\n",
                                "bytes": [123, 10],
                                "logprob": -25.15625,
                            },
                            {
                                "token": "javascript",
                                "bytes": [
                                    106,
                                    97,
                                    118,
                                    97,
                                    115,
                                    99,
                                    114,
                                    105,
                                    112,
                                    116,
                                ],
                                "logprob": -29.75,
                            },
                            {
                                "token": "yaml",
                                "bytes": [121, 97, 109, 108],
                                "logprob": -29.84375,
                            },
                            {
                                "token": "  \n",
                                "bytes": [32, 32, 10],
                                "logprob": -30.15625,
                            },
                            {
                                "token": '{"',
                                "bytes": [123, 34],
                                "logprob": -31.28125,
                            },
                            {
                                "token": "Json",
                                "bytes": [74, 115, 111, 110],
                                "logprob": -31.78125,
                            },
                            {
                                "token": " JSON",
                                "bytes": [32, 74, 83, 79, 78],
                                "logprob": -31.875,
                            },
                            {"token": "\n", "bytes": [10], "logprob": -32.4375},
                            {
                                "token": "j",
                                "bytes": [106],
                                "logprob": -32.46875,
                            },
                            {
                                "token": "    \n",
                                "bytes": [32, 32, 32, 32, 10],
                                "logprob": -33.0,
                            },
                            {
                                "token": ".json",
                                "bytes": [46, 106, 115, 111, 110],
                                "logprob": -33.34375,
                            },
                            {
                                "token": "jos",
                                "bytes": [106, 111, 115],
                                "logprob": -33.40625,
                            },
                            {
                                "token": "\tjson",
                                "bytes": [9, 106, 115, 111, 110],
                                "logprob": -33.84375,
                            },
                            {
                                "token": "   \n",
                                "bytes": [32, 32, 32, 10],
                                "logprob": -34.1875,
                            },
                            {
                                "token": "js",
                                "bytes": [106, 115],
                                "logprob": -34.21875,
                            },
                            {
                                "token": " {\n",
                                "bytes": [32, 123, 10],
                                "logprob": -34.3125,
                            },
                        ],
                    },
                    {
                        "token": "\n",
                        "bytes": [10],
                        "logprob": -9.0883464e-07,
                        "top_logprobs": [
                            {
                                "token": "\n",
                                "bytes": [10],
                                "logprob": -9.0883464e-07,
                            },
                            {
                                "token": " \n",
                                "bytes": [32, 10],
                                "logprob": -14.343751,
                            },
                            {
                                "token": "{\n",
                                "bytes": [123, 10],
                                "logprob": -15.937501,
                            },
                            {
                                "token": "\n\n",
                                "bytes": [10, 10],
                                "logprob": -16.234375,
                            },
                            {
                                "token": "  \n",
                                "bytes": [32, 32, 10],
                                "logprob": -19.328125,
                            },
                            {
                                "token": " {\n",
                                "bytes": [32, 123, 10],
                                "logprob": -20.9375,
                            },
                            {
                                "token": "    \n",
                                "bytes": [32, 32, 32, 32, 10],
                                "logprob": -21.1875,
                            },
                            {
                                "token": "   \n",
                                "bytes": [32, 32, 32, 10],
                                "logprob": -22.625,
                            },
                            {
                                "token": "\t\n",
                                "bytes": [9, 10],
                                "logprob": -23.125,
                            },
                            {
                                "token": "=\n",
                                "bytes": [61, 10],
                                "logprob": -23.234375,
                            },
                            {
                                "token": "\r\n",
                                "bytes": [13, 10],
                                "logprob": -24.0625,
                            },
                            {
                                "token": "\n\n\n",
                                "bytes": [10, 10, 10],
                                "logprob": -24.078125,
                            },
                            {
                                "token": ":\n",
                                "bytes": [58, 10],
                                "logprob": -24.109375,
                            },
                            {
                                "token": "={\n",
                                "bytes": [61, 123, 10],
                                "logprob": -24.4375,
                            },
                            {
                                "token": "        \n",
                                "bytes": [32, 32, 32, 32, 32, 32, 32, 32, 10],
                                "logprob": -24.890625,
                            },
                            {
                                "token": "                                                                ",
                                "bytes": [
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                    32,
                                ],
                                "logprob": -24.984375,
                            },
                            {
                                "token": ".\n",
                                "bytes": [46, 10],
                                "logprob": -25.046875,
                            },
                            {
                                "token": " \n\n",
                                "bytes": [32, 10, 10],
                                "logprob": -25.34375,
                            },
                            {
                                "token": "`\n",
                                "bytes": [96, 10],
                                "logprob": -25.5,
                            },
                            {
                                "token": "     \n",
                                "bytes": [32, 32, 32, 32, 32, 10],
                                "logprob": -25.640625,
                            },
                        ],
                    },
                    {
                        "token": "{\n",
                        "bytes": [123, 10],
                        "logprob": -4.3202e-07,
                        "top_logprobs": [
                            {
                                "token": "{\n",
                                "bytes": [123, 10],
                                "logprob": -4.3202e-07,
                            },
                            {
                                "token": '{"',
                                "bytes": [123, 34],
                                "logprob": -15.84375,
                            },
                            {"token": "{", "bytes": [123], "logprob": -16.0},
                            {
                                "token": " {\n",
                                "bytes": [32, 123, 10],
                                "logprob": -16.4375,
                            },
                            {
                                "token": "{\n\n",
                                "bytes": [123, 10, 10],
                                "logprob": -22.46875,
                            },
                            {"token": " ", "bytes": [32], "logprob": -23.03125},
                            {
                                "token": "   ",
                                "bytes": [32, 32, 32],
                                "logprob": -23.34375,
                            },
                            {
                                "token": "[\n",
                                "bytes": [91, 10],
                                "logprob": -24.65625,
                            },
                            {
                                "token": "{\r\n",
                                "bytes": [123, 13, 10],
                                "logprob": -25.953125,
                            },
                            {
                                "token": "  ",
                                "bytes": [32, 32],
                                "logprob": -26.421875,
                            },
                            {
                                "token": "{\n\n\n",
                                "bytes": [123, 10, 10, 10],
                                "logprob": -28.203125,
                            },
                            {
                                "token": "\n",
                                "bytes": [10],
                                "logprob": -28.453125,
                            },
                            {"token": "\t", "bytes": [9], "logprob": -28.78125},
                            {"token": '"', "bytes": [34], "logprob": -29.21875},
                            {
                                "token": ' {"',
                                "bytes": [32, 123, 34],
                                "logprob": -29.234375,
                            },
                            {
                                "token": "       ",
                                "bytes": [32, 32, 32, 32, 32, 32, 32],
                                "logprob": -29.65625,
                            },
                            {
                                "token": "8",
                                "bytes": [56],
                                "logprob": -29.890625,
                            },
                            {
                                "token": " {",
                                "bytes": [32, 123],
                                "logprob": -29.90625,
                            },
                            {
                                "token": "({\n",
                                "bytes": [40, 123, 10],
                                "logprob": -30.046875,
                            },
                            {
                                "token": ",{\n",
                                "bytes": [44, 123, 10],
                                "logprob": -30.453125,
                            },
                        ],
                    },
                    {
                        "token": " ",
                        "bytes": [32],
                        "logprob": -0.012182552,
                        "top_logprobs": [
                            {
                                "token": " ",
                                "bytes": [32],
                                "logprob": -0.012182552,
                            },
                            {
                                "token": "   ",
                                "bytes": [32, 32, 32],
                                "logprob": -4.4184327,
                            },
                            {
                                "token": "  ",
                                "bytes": [32, 32],
                                "logprob": -10.293432,
                            },
                            {
                                "token": "\t",
                                "bytes": [9],
                                "logprob": -11.168432,
                            },
                            {
                                "token": ' "',
                                "bytes": [32, 34],
                                "logprob": -12.293432,
                            },
                            {
                                "token": '"',
                                "bytes": [34],
                                "logprob": -12.762182,
                            },
                            {
                                "token": "    ",
                                "bytes": [32, 32, 32, 32],
                                "logprob": -19.824682,
                            },
                            {
                                "token": "\xa0",
                                "bytes": [194, 160],
                                "logprob": -20.949682,
                            },
                            {
                                "token": "     ",
                                "bytes": [32, 32, 32, 32, 32],
                                "logprob": -21.324682,
                            },
                            {
                                "token": "       ",
                                "bytes": [32, 32, 32, 32, 32, 32, 32],
                                "logprob": -24.980932,
                            },
                            {
                                "token": "  \n",
                                "bytes": [32, 32, 10],
                                "logprob": -26.230932,
                            },
                            {
                                "token": "\xa0\xa0\xa0",
                                "bytes": [194, 160, 194, 160, 194, 160],
                                "logprob": -26.559057,
                            },
                            {
                                "token": "\t ",
                                "bytes": [9, 32],
                                "logprob": -26.730932,
                            },
                            {
                                "token": "\\xe2\\x80",
                                "bytes": [226, 128],
                                "logprob": -27.605932,
                            },
                            {
                                "token": "    \n",
                                "bytes": [32, 32, 32, 32, 10],
                                "logprob": -27.730932,
                            },
                            {
                                "token": "\u3000",
                                "bytes": [227, 128, 128],
                                "logprob": -27.777807,
                            },
                            {
                                "token": " \xa0",
                                "bytes": [32, 194, 160],
                                "logprob": -28.152807,
                            },
                            {
                                "token": "\xa0\xa0",
                                "bytes": [194, 160, 194, 160],
                                "logprob": -28.465307,
                            },
                            {
                                "token": '"text',
                                "bytes": [34, 116, 101, 120, 116],
                                "logprob": -28.605932,
                            },
                            {
                                "token": '"title',
                                "bytes": [34, 116, 105, 116, 108, 101],
                                "logprob": -28.746557,
                            },
                        ],
                    },
                    {
                        "token": ' "',
                        "bytes": [32, 34],
                        "logprob": 0.0,
                        "top_logprobs": [
                            {"token": ' "', "bytes": [32, 34], "logprob": 0.0},
                            {"token": "\t", "bytes": [9], "logprob": -24.6875},
                            {
                                "token": "\xa0",
                                "bytes": [194, 160],
                                "logprob": -27.25,
                            },
                            {
                                "token": " “",
                                "bytes": [32, 226, 128, 156],
                                "logprob": -28.59375,
                            },
                            {"token": " ", "bytes": [32], "logprob": -28.96875},
                            {
                                "token": " score",
                                "bytes": [32, 115, 99, 111, 114, 101],
                                "logprob": -29.125,
                            },
                            {
                                "token": " '",
                                "bytes": [32, 39],
                                "logprob": -29.375,
                            },
                            {
                                "token": ' "_',
                                "bytes": [32, 34, 95],
                                "logprob": -29.875,
                            },
                            {
                                "token": ' "$',
                                "bytes": [32, 34, 36],
                                "logprob": -32.390625,
                            },
                            {
                                "token": ' "__',
                                "bytes": [32, 34, 95, 95],
                                "logprob": -32.59375,
                            },
                            {
                                "token": ' "\\',
                                "bytes": [32, 34, 92],
                                "logprob": -33.421875,
                            },
                            {
                                "token": '"',
                                "bytes": [34],
                                "logprob": -33.640625,
                            },
                            {
                                "token": ' "/',
                                "bytes": [32, 34, 47],
                                "logprob": -33.859375,
                            },
                            {
                                "token": ' "\n',
                                "bytes": [32, 34, 10],
                                "logprob": -34.078125,
                            },
                            {
                                "token": " Score",
                                "bytes": [32, 83, 99, 111, 114, 101],
                                "logprob": -34.453125,
                            },
                            {
                                "token": ' "@',
                                "bytes": [32, 34, 64],
                                "logprob": -34.953125,
                            },
                            {
                                "token": ' "\\"',
                                "bytes": [32, 34, 92, 34],
                                "logprob": -35.46875,
                            },
                            {
                                "token": ' "<',
                                "bytes": [32, 34, 60],
                                "logprob": -35.515625,
                            },
                            {
                                "token": " {\n",
                                "bytes": [32, 123, 10],
                                "logprob": -35.65625,
                            },
                            {
                                "token": "  ",
                                "bytes": [32, 32],
                                "logprob": -35.859375,
                            },
                        ],
                    },
                    {
                        "token": "score",
                        "bytes": [115, 99, 111, 114, 101],
                        "logprob": 0.0,
                        "top_logprobs": [
                            {
                                "token": "score",
                                "bytes": [115, 99, 111, 114, 101],
                                "logprob": 0.0,
                            },
                            {
                                "token": "Score",
                                "bytes": [83, 99, 111, 114, 101],
                                "logprob": -28.53125,
                            },
                            {
                                "token": "sc",
                                "bytes": [115, 99],
                                "logprob": -33.03125,
                            },
                            {
                                "token": "scores",
                                "bytes": [115, 99, 111, 114, 101, 115],
                                "logprob": -34.21875,
                            },
                            {
                                "token": ".score",
                                "bytes": [46, 115, 99, 111, 114, 101],
                                "logprob": -36.21875,
                            },
                            {
                                "token": "scale",
                                "bytes": [115, 99, 97, 108, 101],
                                "logprob": -36.28125,
                            },
                            {
                                "token": " score",
                                "bytes": [32, 115, 99, 111, 114, 101],
                                "logprob": -36.34375,
                            },
                            {
                                "token": "s",
                                "bytes": [115],
                                "logprob": -36.40625,
                            },
                            {
                                "token": "SCORE",
                                "bytes": [83, 67, 79, 82, 69],
                                "logprob": -37.09375,
                            },
                            {
                                "token": "\tscore",
                                "bytes": [9, 115, 99, 111, 114, 101],
                                "logprob": -37.875,
                            },
                            {
                                "token": "_score",
                                "bytes": [95, 115, 99, 111, 114, 101],
                                "logprob": -38.09375,
                            },
                            {
                                "token": "scope",
                                "bytes": [115, 99, 111, 112, 101],
                                "logprob": -38.71875,
                            },
                            {
                                "token": "(score",
                                "bytes": [40, 115, 99, 111, 114, 101],
                                "logprob": -39.03125,
                            },
                            {
                                "token": "source",
                                "bytes": [115, 111, 117, 114, 99, 101],
                                "logprob": -39.53125,
                            },
                            {
                                "token": "scan",
                                "bytes": [115, 99, 97, 110],
                                "logprob": -40.28125,
                            },
                            {
                                "token": "-score",
                                "bytes": [45, 115, 99, 111, 114, 101],
                                "logprob": -40.34375,
                            },
                            {
                                "token": "soc",
                                "bytes": [115, 111, 99],
                                "logprob": -41.21875,
                            },
                            {
                                "token": "core",
                                "bytes": [99, 111, 114, 101],
                                "logprob": -41.375,
                            },
                            {
                                "token": "sample",
                                "bytes": [115, 97, 109, 112, 108, 101],
                                "logprob": -41.375,
                            },
                            {
                                "token": "reason",
                                "bytes": [114, 101, 97, 115, 111, 110],
                                "logprob": -41.65625,
                            },
                        ],
                    },
                    {
                        "token": '":',
                        "bytes": [34, 58],
                        "logprob": 0.0,
                        "top_logprobs": [
                            {"token": '":', "bytes": [34, 58], "logprob": 0.0},
                            {"token": '"', "bytes": [34], "logprob": -19.71875},
                            {
                                "token": "':",
                                "bytes": [39, 58],
                                "logprob": -26.96875,
                            },
                            {
                                "token": "”:",
                                "bytes": [226, 128, 157, 58],
                                "logprob": -27.46875,
                            },
                            {"token": ":", "bytes": [58], "logprob": -27.625},
                            {
                                "token": '":\n',
                                "bytes": [34, 58, 10],
                                "logprob": -29.21875,
                            },
                            {
                                "token": '":"',
                                "bytes": [34, 58, 34],
                                "logprob": -29.84375,
                            },
                            {
                                "token": '\\":',
                                "bytes": [92, 34, 58],
                                "logprob": -32.15625,
                            },
                            {
                                "token": '",',
                                "bytes": [34, 44],
                                "logprob": -33.5625,
                            },
                            {
                                "token": '";',
                                "bytes": [34, 59],
                                "logprob": -33.796875,
                            },
                            {
                                "token": '"":',
                                "bytes": [34, 34, 58],
                                "logprob": -34.359375,
                            },
                            {
                                "token": ' ":',
                                "bytes": [32, 34, 58],
                                "logprob": -34.4375,
                            },
                            {
                                "token": '":-',
                                "bytes": [34, 58, 45],
                                "logprob": -35.5625,
                            },
                            {
                                "token": '":\n\n',
                                "bytes": [34, 58, 10, 10],
                                "logprob": -36.75,
                            },
                            {
                                "token": '":[',
                                "bytes": [34, 58, 91],
                                "logprob": -36.8125,
                            },
                            {
                                "token": '"):',
                                "bytes": [34, 41, 58],
                                "logprob": -37.5625,
                            },
                            {
                                "token": '"]:',
                                "bytes": [34, 93, 58],
                                "logprob": -37.625,
                            },
                            {
                                "token": "<|end|>",
                                "bytes": None,
                                "logprob": -37.6875,
                            },
                            {
                                "token": "):",
                                "bytes": [41, 58],
                                "logprob": -38.203125,
                            },
                            {
                                "token": ':"',
                                "bytes": [58, 34],
                                "logprob": -38.265625,
                            },
                        ],
                    },
                    {
                        "token": " ",
                        "bytes": [32],
                        "logprob": -1.2664457e-06,
                        "top_logprobs": [
                            {
                                "token": " ",
                                "bytes": [32],
                                "logprob": -1.2664457e-06,
                            },
                            {
                                "token": "9",
                                "bytes": [57],
                                "logprob": -13.718751,
                            },
                            {
                                "token": "8",
                                "bytes": [56],
                                "logprob": -15.609376,
                            },
                            {
                                "token": "10",
                                "bytes": [49, 48],
                                "logprob": -18.796877,
                            },
                            {
                                "token": ' "',
                                "bytes": [32, 34],
                                "logprob": -20.437502,
                            },
                            {
                                "token": "7",
                                "bytes": [55],
                                "logprob": -20.593752,
                            },
                            {
                                "token": " \n",
                                "bytes": [32, 10],
                                "logprob": -21.781252,
                            },
                            {
                                "token": "  ",
                                "bytes": [32, 32],
                                "logprob": -22.281252,
                            },
                            {
                                "token": "\xa0",
                                "bytes": [194, 160],
                                "logprob": -23.265627,
                            },
                            {
                                "token": "\t",
                                "bytes": [9],
                                "logprob": -23.984377,
                            },
                            {
                                "token": "   ",
                                "bytes": [32, 32, 32],
                                "logprob": -24.296877,
                            },
                            {
                                "token": " -",
                                "bytes": [32, 45],
                                "logprob": -25.140627,
                            },
                            {
                                "token": "<|end|>",
                                "bytes": None,
                                "logprob": -25.156252,
                            },
                            {
                                "token": " \n\n",
                                "bytes": [32, 10, 10],
                                "logprob": -25.726564,
                            },
                            {
                                "token": "6",
                                "bytes": [54],
                                "logprob": -26.062502,
                            },
                            {"token": ",", "bytes": [44], "logprob": -26.42969},
                            {
                                "token": " .",
                                "bytes": [32, 46],
                                "logprob": -26.445314,
                            },
                            {
                                "token": "  \n",
                                "bytes": [32, 32, 10],
                                "logprob": -26.507814,
                            },
                            {
                                "token": ":",
                                "bytes": [58],
                                "logprob": -26.570314,
                            },
                            {"token": "-", "bytes": [45], "logprob": -26.64844},
                        ],
                    },
                    {
                        "token": "9",
                        "bytes": [57],
                        "logprob": -0.11450484,
                        "top_logprobs": [
                            {
                                "token": "9",
                                "bytes": [57],
                                "logprob": -0.11450484,
                            },
                            {
                                "token": "8",
                                "bytes": [56],
                                "logprob": -2.2395048,
                            },
                            {
                                "token": "10",
                                "bytes": [49, 48],
                                "logprob": -6.44263,
                            },
                            {"token": "7", "bytes": [55], "logprob": -9.333255},
                            {
                                "token": " ",
                                "bytes": [32],
                                "logprob": -14.708255,
                            },
                            {
                                "token": "6",
                                "bytes": [54],
                                "logprob": -17.083256,
                            },
                            {"token": "5", "bytes": [53], "logprob": -20.16138},
                            {
                                "token": "09",
                                "bytes": [48, 57],
                                "logprob": -20.94263,
                            },
                            {
                                "token": "0",
                                "bytes": [48],
                                "logprob": -21.489506,
                            },
                            {
                                "token": " nine",
                                "bytes": [32, 110, 105, 110, 101],
                                "logprob": -21.794193,
                            },
                            {
                                "token": "08",
                                "bytes": [48, 56],
                                "logprob": -22.427006,
                            },
                            {
                                "token": "4",
                                "bytes": [52],
                                "logprob": -22.591068,
                            },
                            {
                                "token": "９",
                                "bytes": [239, 188, 153],
                                "logprob": -22.614506,
                            },
                            {
                                "token": "<|end|>",
                                "bytes": None,
                                "logprob": -23.044193,
                            },
                            {
                                "token": "\xa0",
                                "bytes": [194, 160],
                                "logprob": -23.075443,
                            },
                            {
                                "token": "90",
                                "bytes": [57, 48],
                                "logprob": -23.497318,
                            },
                            {
                                "token": "3",
                                "bytes": [51],
                                "logprob": -23.606693,
                            },
                            {
                                "token": " eight",
                                "bytes": [32, 101, 105, 103, 104, 116],
                                "logprob": -23.895756,
                            },
                            {
                                "token": "８",
                                "bytes": [239, 188, 152],
                                "logprob": -24.419193,
                            },
                            {
                                "token": "\n\n",
                                "bytes": [10, 10],
                                "logprob": -24.450443,
                            },
                        ],
                    },
                    {
                        "token": ",\n",
                        "bytes": [44, 10],
                        "logprob": -4.5491004e-05,
                        "top_logprobs": [
                            {
                                "token": ",\n",
                                "bytes": [44, 10],
                                "logprob": -4.5491004e-05,
                            },
                            {
                                "token": ",",
                                "bytes": [44],
                                "logprob": -10.000046,
                            },
                            {
                                "token": ".",
                                "bytes": [46],
                                "logprob": -18.593796,
                            },
                            {
                                "token": " ,\n",
                                "bytes": [32, 44, 10],
                                "logprob": -20.32817,
                            },
                            {
                                "token": ",\n\n",
                                "bytes": [44, 10, 10],
                                "logprob": -20.79692,
                            },
                            {
                                "token": "\n",
                                "bytes": [10],
                                "logprob": -26.656296,
                            },
                            {
                                "token": " ,",
                                "bytes": [32, 44],
                                "logprob": -27.281296,
                            },
                            {
                                "token": ".\n",
                                "bytes": [46, 10],
                                "logprob": -27.45317,
                            },
                            {
                                "token": '",\n',
                                "bytes": [34, 44, 10],
                                "logprob": -27.54692,
                            },
                            {
                                "token": ',"',
                                "bytes": [44, 34],
                                "logprob": -27.64067,
                            },
                            {
                                "token": ",\n\n\n",
                                "bytes": [44, 10, 10, 10],
                                "logprob": -27.64067,
                            },
                            {
                                "token": ",\r\n",
                                "bytes": [44, 13, 10],
                                "logprob": -28.26567,
                            },
                            {"token": " ", "bytes": [32], "logprob": -28.70317},
                            {
                                "token": ";\n",
                                "bytes": [59, 10],
                                "logprob": -28.70317,
                            },
                            {
                                "token": ".,\n",
                                "bytes": [46, 44, 10],
                                "logprob": -28.92192,
                            },
                            {
                                "token": ":\n",
                                "bytes": [58, 10],
                                "logprob": -30.093796,
                            },
                            {
                                "token": ",\\\n",
                                "bytes": [44, 92, 10],
                                "logprob": -30.98442,
                            },
                            {
                                "token": "(),\n",
                                "bytes": [40, 41, 44, 10],
                                "logprob": -31.312546,
                            },
                            {"token": "/", "bytes": [47], "logprob": -31.51567},
                            {
                                "token": "，\n",
                                "bytes": [239, 188, 140, 10],
                                "logprob": -31.51567,
                            },
                        ],
                    },
                ]
            },
        },
    )
    assert round(
        metric.generate_logprobs_based_score(9, raw_response), 8
    ) == round(0.8893309402242041 * 10, 8)
