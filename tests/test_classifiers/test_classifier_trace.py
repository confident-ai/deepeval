import json

from deepeval.classifiers import Classifier
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, ToolCall


class ReplyModel(DeepEvalBaseLLM):
    def __init__(self):
        self.prompts = []

    def load_model(self):
        return self

    def generate(self, prompt, schema=None):
        self.prompts.append(prompt)
        return json.dumps({"label": "pricing", "reason": "Asked for a price."})

    async def a_generate(self, prompt, schema=None):
        return self.generate(prompt, schema)

    def get_model_name(self):
        return "reply-model"


def _classify(test_case):
    model = ReplyModel()
    classifier = Classifier(
        name="topic",
        labels=["pricing", "support"],
        model=model,
        async_mode=False,
    )
    assert classifier.classify(test_case, _show_indicator=False) == "pricing"
    return model.prompts[0]


def _trace_json(prompt):
    return json.loads(prompt.split("Trace:\n", 1)[1].split("\n\n---", 1)[0])


def test_test_case_without_trace_keeps_single_turn_prompt():
    prompt = _classify(
        LLMTestCase(input="How much is Pro?", actual_output="$20.")
    )
    assert "Test Case:" in prompt and "Trace:" not in prompt


def test_trace_is_shown_with_flattened_spans():
    test_case = LLMTestCase(
        input="How much is Pro?",
        actual_output="$20.",
        tools_called=[ToolCall(name="lookup_price")],
    )
    test_case._trace_dict = {
        "name": "agent",
        "type": "agent",
        "input": "How much is Pro?",
        "children": [
            {
                "name": "lookup_price",
                "type": "tool",
                "output": "20",
                "children": [],
            },
            {"name": "llm", "type": "llm", "error": "timeout", "children": []},
        ],
    }
    prompt = _classify(test_case)
    assert "Given a trace from an AI application" in prompt
    trace = _trace_json(prompt)
    assert trace["input"] == "How much is Pro?" and trace["output"] == "$20."
    assert trace["toolsCalled"][0]["name"] == "lookup_price"
    assert [(s["index"], s["name"], s["status"]) for s in trace["spans"]] == [
        (1, "agent", "SUCCESS"),
        (2, "lookup_price", "SUCCESS"),
        (3, "llm", "ERRORED"),
    ]
    assert trace["spans"][2]["error"] == "timeout"
