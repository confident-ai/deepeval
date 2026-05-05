import pytest

from deepeval import assert_test
from deepeval.dataset import EvaluationDataset
from deepeval.test_case import LLMTestCase, ToolCall


DATASET_PATH = "tests/evals/.dataset.json"
EVALUATION_MODEL = "EVALUATION_MODEL"

# Replace with DeepEval metric instances, reusing existing project metrics first.
END_TO_END_METRICS = []

dataset = EvaluationDataset()
dataset.add_goldens_from_json_file(file_path=DATASET_PATH)


def TARGET_APP_ENTRYPOINT(user_input):
    raise NotImplementedError("Replace TARGET_APP_ENTRYPOINT with your app.")


def APP_RESPONSE_ADAPTER(response):
    """Return fields needed for LLMTestCase from the app response."""
    return {
        "actual_output": response,
        "retrieval_context": None,
        "tools_called": None,
    }


def to_deepeval_tool_calls(raw_tool_calls):
    return [
        ToolCall(
            name=tool_call["name"],
            input_parameters=tool_call.get("input_parameters"),
            output=tool_call.get("output"),
        )
        for tool_call in raw_tool_calls or []
    ]


@pytest.mark.parametrize("golden", dataset.goldens)
def test_single_turn(golden):
    response = TARGET_APP_ENTRYPOINT(golden.input)
    fields = APP_RESPONSE_ADAPTER(response)

    test_case = LLMTestCase(
        input=golden.input,
        actual_output=fields["actual_output"],
        expected_output=getattr(golden, "expected_output", None),
        context=getattr(golden, "context", None),
        retrieval_context=fields.get("retrieval_context"),
        tools_called=to_deepeval_tool_calls(fields.get("tools_called")),
        expected_tools=getattr(golden, "expected_tools", None),
    )

    assert_test(test_case=test_case, metrics=END_TO_END_METRICS)
