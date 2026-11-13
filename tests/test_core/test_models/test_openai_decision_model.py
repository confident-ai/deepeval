import asyncio

import openai
import pytest

from deepeval.errors import DeepEvalError
from deepeval.metrics.utils.decision import (
    _is_permanent,
    _system_one_error_types,
)
from deepeval.models.system_one import (
    ChoiceQuestion,
    NoulQuestion,
    OpenAIDecisionModel,
    ScoreQuestion,
    context_limits,
)
from deepeval.models.system_one.constants import (
    OPENAI_DECISIONS_MAX_QUESTIONS,
)


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.bodies = []

    def post(self, path, *, body, cast_to):
        assert path == "/decisions"
        assert cast_to is object
        self.bodies.append(body)
        return self.responses.pop(0)


class FakeAsyncClient(FakeClient):
    async def post(self, path, *, body, cast_to):
        return FakeClient.post(self, path, body=body, cast_to=cast_to)


def _model(client):
    model = OpenAIDecisionModel(api_key="test-key")
    model.load_model = lambda async_mode=False: client
    return model


def _usage(input_tokens):
    return {"input_tokens": input_tokens, "output_tokens": 0}


QUESTIONS = {
    "relevant": NoulQuestion(
        instructions="Is the output relevant?",
        true="It answers the question.",
        false="It does not.",
    ),
    "tone": ChoiceQuestion(
        instructions="What tone is the output?",
        options={"formal": "Formal register.", "casual": None},
    ),
    "quality": ScoreQuestion(
        instructions="Rate the quality.", levels=["bad", "ok", "great"]
    ),
}

ANSWERS = {
    "answers": [
        {"type": "predicate", "name": "relevant", "probability": 0.9},
        {
            "type": "choice",
            "name": "tone",
            "choice": "formal",
            "confidence": 0.8,
            "probabilities": [
                {"value": "formal", "probability": 0.85},
                {"value": "casual", "probability": 0.15},
            ],
        },
        {
            "type": "score",
            "name": "quality",
            "score": 1.6,
            "confidence": 0.5,
            "probabilities": [
                {"value": 0, "label": "level_0", "probability": 0.1},
                {"value": 1, "label": "level_1", "probability": 0.2},
                {"value": 2, "label": "level_2", "probability": 0.7},
            ],
        },
    ],
    "model": "gpt-6-luna",
    "usage": _usage(1000),
}


def test_translates_questions_and_state():
    client = FakeClient([ANSWERS])
    _model(client).decide({"input": "hi"}, QUESTIONS)

    body = client.bodies[0]
    assert body["model"] == "gpt-6-luna"
    assert body["input"] == [
        {
            "role": "user",
            "type": "message",
            "content": [{"type": "input_json", "data": {"input": "hi"}}],
        }
    ]
    predicate, choice, score = body["questions"]
    assert predicate == {
        "type": "predicate",
        "name": "relevant",
        "instructions": "Is the output relevant?\n\nTrue when: It answers "
        "the question.\nFalse when: It does not.",
    }
    assert choice["choices"] == [
        {"value": "formal", "description": "Formal register."},
        {"value": "casual"},
    ]
    assert score["levels"] == [
        {"label": "level_0", "description": "bad"},
        {"label": "level_1", "description": "ok"},
        {"label": "level_2", "description": "great"},
    ]


def test_string_state_is_sent_as_text():
    client = FakeClient([ANSWERS])
    _model(client).decide("plain text", QUESTIONS)
    assert client.bodies[0]["input"] == "plain text"


def _nested(depth):
    state = {}
    for _ in range(depth - 1):
        state = {"child": state}
    return state


@pytest.mark.parametrize(
    "state", [{"spans": list(range(10_000))}, _nested(33)]
)
def test_json_over_openai_limits_is_sent_as_text(state):
    client = FakeClient([ANSWERS])
    _model(client).decide(state, QUESTIONS)
    assert isinstance(client.bodies[0]["input"], str)


def test_parses_answers_and_cost():
    answers, cost = _model(FakeClient([ANSWERS])).decide("x", QUESTIONS)

    assert answers.nouls["relevant"].probability == 0.9
    assert answers.choices["tone"].choice == "formal"
    assert answers.choices["tone"].probabilities == {
        "formal": 0.85,
        "casual": 0.15,
    }
    assert answers.scores["quality"].probabilities == {0: 0.1, 1: 0.2, 2: 0.7}
    assert answers.scores["quality"].normalized == pytest.approx(0.8)
    assert cost == pytest.approx(1000 * 0.10 / 1e6)
    assert cost.input_tokens == 1000


def test_refusal_raises():
    refused = {
        "answers": [{"type": "refusal", "name": "relevant"}],
        "usage": _usage(10),
    }
    with pytest.raises(DeepEvalError, match="refused"):
        _model(FakeClient([refused])).decide(
            "x", {"relevant": QUESTIONS["relevant"]}
        )


def test_missing_answer_raises():
    empty = {"answers": [], "usage": _usage(10)}
    with pytest.raises(DeepEvalError, match="no answer"):
        _model(FakeClient([empty])).decide(
            "x", {"relevant": QUESTIONS["relevant"]}
        )


def test_splits_requests_over_the_question_cap():
    count = OPENAI_DECISIONS_MAX_QUESTIONS + 1
    questions = {f"q{i}": NoulQuestion(instructions="?") for i in range(count)}

    def response(names):
        return {
            "answers": [
                {"type": "predicate", "name": n, "probability": 0.5}
                for n in names
            ],
            "usage": _usage(5),
        }

    names = list(questions)
    client = FakeAsyncClient(
        [
            response(names[:OPENAI_DECISIONS_MAX_QUESTIONS]),
            response(names[OPENAI_DECISIONS_MAX_QUESTIONS:]),
        ]
    )
    answers, cost = asyncio.run(_model(client).a_decide("x", questions))

    assert [len(b["questions"]) for b in client.bodies] == [
        OPENAI_DECISIONS_MAX_QUESTIONS,
        1,
    ]
    assert len(answers.nouls) == count
    assert cost.input_tokens == 10


def test_name_and_context_limits():
    model = OpenAIDecisionModel(api_key="test-key")
    assert model.get_model_name() == "gpt-6-luna (OpenAI)"
    assert context_limits(model) == {
        "max_state_tokens": None,
        "max_request_tokens": None,
    }


def test_openai_errors_take_the_system_one_fallback():
    assert openai.APIError in _system_one_error_types()
    auth_error = openai.AuthenticationError.__new__(openai.AuthenticationError)
    assert _is_permanent(auth_error)
