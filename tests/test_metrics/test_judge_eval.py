import json

import pytest

from deepeval.metrics import JudgeEval
from deepeval.metrics.g_eval.schema import ReasonScore
from deepeval.metrics.judge_eval import (
    JudgeEvalField,
    JudgeEvalMessage,
    JudgeEvalRole,
    JudgeEvalVariable,
)
from deepeval.metrics.judge_eval.utils import (
    render_messages,
    resolve_path,
    resolve_variables,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, ToolCall


INPUT = {
    "messages": [
        {"role": "system", "content": "You are a payroll assistant."},
        {"role": "user", "content": "Where do I add a super fund?"},
    ]
}
OUTPUT = {
    "messages": [
        {"role": "assistant", "content": "Go to Employees > Super."},
    ]
}


class RecordingJudge(DeepEvalBaseLLM):
    def __init__(self, score: int = 8):
        self.prompts = []
        self.score = score
        super().__init__(model="recording-judge")

    def load_model(self, *args, **kwargs):
        return None

    def generate(self, prompt, schema=None, **kwargs):
        self.prompts.append(prompt)
        return ReasonScore(reason="Grounded answer.", score=self.score)

    async def a_generate(self, prompt, schema=None, **kwargs):
        return self.generate(prompt, schema=schema, **kwargs)

    def get_model_name(self, *args, **kwargs):
        return "recording-judge"


def _messages():
    return [
        JudgeEvalMessage(
            role=JudgeEvalRole.SYSTEM, content="You grade payroll answers."
        ),
        JudgeEvalMessage(
            role=JudgeEvalRole.USER,
            content="QUESTION: {{question}}\nANSWER: {{ answer }}",
        ),
    ]


def _variables():
    return {
        "question": JudgeEvalVariable(
            field=JudgeEvalField.INPUT, path=["messages", -1, "content"]
        ),
        "answer": JudgeEvalVariable(
            field=JudgeEvalField.OUTPUT, path=["messages", 0, "content"]
        ),
    }


def _test_case():
    return LLMTestCase(
        input=json.dumps(INPUT),
        actual_output=json.dumps(OUTPUT),
        metadata={"tenant": {"region": "au"}},
        tools_called=[
            ToolCall(name="search_docs", input_parameters={"q": "super"})
        ],
    )


def test_resolve_path_supports_keys_indexes_last_and_all():
    assert resolve_path(INPUT, ["messages", 0, "role"]) == "system"
    assert resolve_path(INPUT, ["messages", -1, "content"]) == (
        "Where do I add a super fund?"
    )
    assert resolve_path(INPUT, ["messages", "*", "role"]) == ["system", "user"]
    assert resolve_path(INPUT, []) == INPUT


def test_resolve_path_returns_none_on_miss():
    assert resolve_path(INPUT, ["messages", 5, "content"]) is None
    assert resolve_path(INPUT, ["missing"]) is None
    assert resolve_path(INPUT, ["messages", True]) is None
    assert resolve_path("plain text", ["messages"]) is None


def test_resolve_path_parses_nested_json_strings():
    value = {"payload": json.dumps({"answer": "42"})}
    assert resolve_path(value, ["payload", "answer"]) == "42"


def test_resolve_variables_reads_every_field():
    variables = {
        "question": JudgeEvalVariable(
            field=JudgeEvalField.INPUT, path=["messages", 1, "content"]
        ),
        "region": JudgeEvalVariable(
            field=JudgeEvalField.METADATA, path=["tenant", "region"]
        ),
        "tools": JudgeEvalVariable(
            field=JudgeEvalField.TOOLS_CALLED, path=["*", "name"]
        ),
        "whole_output": JudgeEvalVariable(field=JudgeEvalField.OUTPUT),
        "missing": JudgeEvalVariable(
            field=JudgeEvalField.OUTPUT, path=["messages", 9]
        ),
    }

    values = resolve_variables(_test_case(), variables)

    assert values["question"] == "Where do I add a super fund?"
    assert values["region"] == "au"
    assert values["tools"] == '["search_docs"]'
    assert json.loads(values["whole_output"]) == OUTPUT
    assert values["missing"] == ""


def test_plain_text_input_is_kept_as_is():
    test_case = LLMTestCase(input="hello", actual_output="hi")
    values = resolve_variables(
        test_case,
        {"question": JudgeEvalVariable(field=JudgeEvalField.INPUT)},
    )
    assert values["question"] == "hello"


def test_render_messages_labels_roles_and_fills_variables():
    rendered = render_messages(_messages(), {"question": "Q?", "answer": "A."})
    assert rendered == (
        "System:\nYou grade payroll answers.\n\n"
        "User:\nQUESTION: Q?\nANSWER: A."
    )


def test_unmapped_variables_are_rejected():
    with pytest.raises(ValueError, match="answer"):
        JudgeEval(
            name="Decline Quality",
            messages=_messages(),
            variables={"question": _variables()["question"]},
            model=RecordingJudge(),
        )


def test_system_message_must_come_first():
    with pytest.raises(ValueError, match="system"):
        JudgeEval(
            name="Decline Quality",
            messages=list(reversed(_messages())),
            variables=_variables(),
            model=RecordingJudge(),
        )


@pytest.mark.parametrize("score_range", [(5, 5), (10, 1)])
def test_invalid_score_range_is_rejected(score_range):
    with pytest.raises(ValueError, match="score_range"):
        JudgeEval(
            name="Decline Quality",
            messages=_messages(),
            variables=_variables(),
            score_range=score_range,
            model=RecordingJudge(),
        )


@pytest.mark.parametrize("async_mode", [True, False])
def test_measure_scales_score_to_unit_range(async_mode):
    judge = RecordingJudge(score=4)
    metric = JudgeEval(
        name="Decline Quality",
        messages=_messages(),
        variables=_variables(),
        score_range=(1, 5),
        model=judge,
        async_mode=async_mode,
    )

    score = metric.measure(_test_case())

    assert score == pytest.approx(0.75)
    assert metric.success is True
    assert metric.reason == "Grounded answer."
    assert "QUESTION: Where do I add a super fund?" in judge.prompts[0]
    assert "ANSWER: Go to Employees > Super." in judge.prompts[0]
    assert "between 1 and 5" in judge.prompts[0]


def test_strict_mode_only_passes_on_max_score():
    metric = JudgeEval(
        name="Decline Quality",
        messages=_messages(),
        variables=_variables(),
        score_range=(0, 10),
        model=RecordingJudge(score=9),
        strict_mode=True,
        async_mode=False,
    )

    assert metric.measure(_test_case()) == 0
    assert metric.success is False


def test_name_has_judge_eval_suffix():
    metric = JudgeEval(
        name="Decline Quality",
        messages=_messages(),
        variables=_variables(),
        model=RecordingJudge(),
    )
    assert metric.__name__ == "Decline Quality [JudgeEval]"
