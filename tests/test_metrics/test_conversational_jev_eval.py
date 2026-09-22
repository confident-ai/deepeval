"""ConversationalJevEval: the multi-turn counterpart of JevEval, against a fake
System One model."""

import re
from typing import Any, Dict, Optional, Tuple

import pytest

from deepeval.metrics import ConversationalJevEval
from deepeval.metrics.jev_eval import Choice, Noul, Score
from deepeval.metrics.jev_eval.utils import construct_multi_turn_state
from deepeval.models import DeepEvalBaseLLM, DeepEvalBaseSystemOneModel
from deepeval.models.system_one.schema import (
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
    SystemOneAnswers,
)
from deepeval.test_case import ConversationalTestCase, MultiTurnParams, Turn


class FakeSystemOneModel(DeepEvalBaseSystemOneModel):
    def __init__(self, answers: SystemOneAnswers):
        super().__init__("fake-jev")
        self.answers = answers
        self.calls = []

    def load_model(self, *args, **kwargs):
        return None

    def get_model_name(self):
        return "fake-jev"

    def decide(
        self, state: Any, questions: Dict[str, Any]
    ) -> Tuple[SystemOneAnswers, Optional[float]]:
        self.calls.append((state, questions))
        return self.answers, 0.0

    async def a_decide(self, state, questions):
        return self.decide(state, questions)


class CannedLLM(DeepEvalBaseLLM):
    def __init__(self, reply: str):
        super().__init__("canned-llm")
        self.reply = reply
        self.prompts = []

    def load_model(self, *args, **kwargs):
        return None

    def get_model_name(self):
        return "canned-llm"

    def generate(self, prompt, *args, **kwargs):
        self.prompts.append(prompt)
        return self.reply

    async def a_generate(self, prompt, *args, **kwargs):
        return self.generate(prompt)


TEST_CASE = ConversationalTestCase(
    scenario="A customer asks for a refund on a late order.",
    expected_outcome="The refund amount is confirmed.",
    turns=[
        Turn(
            role="user",
            content="My order arrived a week late. I want a refund.",
        ),
        Turn(
            role="assistant",
            content="Sorry about that. What is your order number?",
        ),
        Turn(role="user", content="It's 4471. How much do I get back?"),
        Turn(
            role="assistant",
            content="You'll receive a refund within 5-7 days.",
        ),
        Turn(role="user", content="ok thanks"),
    ],
)

QUESTIONS = [
    Noul("The assistant verifies the order before acting on the refund."),
    Score(
        "How fully is the user's request settled by the end of turns?",
        levels=["Not addressed", "Partly settled", "Fully resolved"],
    ),
    Choice(
        "What did the assistant do about the amount the user asked for?",
        options={
            "stated_it": 1.0,
            "deferred_it": 0.5,
            "ignored_it": 0.0,
            "never_asked": None,
        },
    ),
]

ANSWERS = SystemOneAnswers(
    nouls={"q_0": NoulAnswer(probability=0.9)},
    scores={
        "q_1": ScoreAnswer(
            score=1.1, probabilities={0: 0.1, 1: 0.7, 2: 0.2}, confidence=0.7
        )
    },
    choices={
        "q_2": ChoiceAnswer(
            choice="ignored_it",
            probabilities={
                "stated_it": 0.05,
                "deferred_it": 0.25,
                "ignored_it": 0.65,
                "never_asked": 0.05,
            },
            confidence=0.65,
        )
    },
)


def make_metric(**kwargs) -> ConversationalJevEval:
    kwargs.setdefault("include_reason", False)
    return ConversationalJevEval(
        name="Refund Handling",
        evaluation_params=[
            MultiTurnParams.SCENARIO,
            MultiTurnParams.EXPECTED_OUTCOME,
        ],
        questions=QUESTIONS,
        system_one_model=FakeSystemOneModel(ANSWERS),
        async_mode=False,
        **kwargs,
    )


def test_content_and_role_are_always_included():
    metric = make_metric()
    assert MultiTurnParams.CONTENT in metric.evaluation_params
    assert MultiTurnParams.ROLE in metric.evaluation_params


def test_state_shape():
    state = construct_multi_turn_state(
        [
            MultiTurnParams.SCENARIO,
            MultiTurnParams.EXPECTED_OUTCOME,
            MultiTurnParams.CONTENT,
            MultiTurnParams.ROLE,
        ],
        TEST_CASE,
    )
    assert set(state) == {"turns", "test_case"}
    assert len(state["turns"]) == 5
    assert state["turns"][1] == {
        "role": "assistant",
        "content": "Sorry about that. What is your order number?",
    }
    assert state["test_case"] == {
        "scenario": TEST_CASE.scenario,
        "expected_outcome": TEST_CASE.expected_outcome,
    }


def test_measure_end_to_end():
    metric = make_metric()
    score = metric.measure(TEST_CASE)
    # noul 0.9, score 1.1/2 = 0.55, choice (0.05*1 + 0.25*0.5 + 0.65*0)/0.95
    choice_v = (0.05 * 1.0 + 0.25 * 0.5) / 0.95
    assert score == pytest.approx((0.9 + 0.55 + choice_v) / 3)
    assert metric.success is True
    assert len(metric.score_breakdown) == 3
    assert metric.system_one_model.calls[0][0]["turns"][0]["role"] == "user"
    assert metric.__name__ == "Refund Handling [Conversational JevEval]"


def test_reason_prompt_cites_turns_and_hides_numbers():
    llm = CannedLLM('{"reason": "ok"}')
    metric = make_metric(include_reason=True, model=llm)
    metric.measure(TEST_CASE)
    prompt = llm.prompts[0]
    for q in QUESTIONS:
        assert q.text in prompt
    assert "clearly holds" in prompt
    assert '"Partly settled"' in prompt
    assert '"ignored_it"' in prompt
    assert "What is your order number?" in prompt
    assert TEST_CASE.scenario in prompt
    assert not re.search(r"\b0\.\d+", prompt)
    assert "%" not in prompt


def test_missing_scenario_raises():
    metric = make_metric()
    with pytest.raises(Exception):
        metric.measure(
            ConversationalTestCase(
                turns=[Turn(role="user", content="hi")],
            )
        )
