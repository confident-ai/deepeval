from unittest.mock import patch

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.test_case import ToolCall
from deepeval.utils import convert_keys_to_snake_case


def test_convert_keys_to_snake_case_keeps_user_supplied_keys():
    data = {
        "expectedOutput": "x",
        "expectedTools": [
            {
                "name": "ask_question",
                "inputParameters": {"repoName": "confident-ai/deepeval"},
                "output": {"answerText": "y"},
            }
        ],
        "customColumnKeyValues": {"userSegment": "enterprise"},
        "expectedLabels": {"toneCheck": "on_tone"},
        "additionalMetadata": {"traceId": 1},
    }

    converted = convert_keys_to_snake_case(data)

    assert converted["expected_output"] == "x"
    tool = converted["expected_tools"][0]
    assert tool["input_parameters"] == {"repoName": "confident-ai/deepeval"}
    assert tool["output"] == {"answerText": "y"}
    assert converted["custom_column_key_values"] == {
        "userSegment": "enterprise"
    }
    assert converted["expected_labels"] == {"toneCheck": "on_tone"}
    assert converted["additional_metadata"] == {"traceId": 1}


def test_pull_returns_the_goldens_that_were_pushed():
    golden = Golden(
        input="What does the repo use for metrics?",
        expected_tools=[
            ToolCall(
                name="ask_question",
                input_parameters={"repoName": "confident-ai/deepeval"},
                output={"answerText": "deepeval"},
            )
        ],
        custom_column_key_values={"userSegment": "enterprise"},
        expected_labels={"toneCheck": "on_tone"},
    )
    stored = {}

    def fake_send_request(self, method, endpoint, body=None, **kwargs):
        if body is not None:
            stored["goldens"] = body["goldens"]
            return None, None
        return {"id": "dataset-id", "goldens": stored["goldens"]}, None

    with patch(
        "deepeval.dataset.dataset.Api.__init__", lambda self, **kwargs: None
    ), patch("deepeval.dataset.dataset.Api.send_request", fake_send_request):
        EvaluationDataset(goldens=[golden]).push("alias")
        pulled = EvaluationDataset()
        pulled.pull("alias")

    pulled_golden = pulled.goldens[0]
    assert pulled_golden.expected_tools == golden.expected_tools
    assert pulled_golden.expected_tools[0].input_parameters == {
        "repoName": "confident-ai/deepeval"
    }
    assert pulled_golden.custom_column_key_values == {
        "userSegment": "enterprise"
    }
    assert pulled_golden.expected_labels == {"toneCheck": "on_tone"}
