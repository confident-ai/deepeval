"""JevEval: Jev-native score metric. All tests run against a fake System One
model, so no network or API key is needed. The one live-LLM reason test is
skipped without OPENAI_API_KEY."""

import os
import re
from typing import Any, Dict, Optional, Tuple

import pytest

from deepeval.metrics import JevEval
from deepeval.metrics.jev_eval import Choice, Noul, Score
from deepeval.metrics.jev_eval.utils import (
    aggregate,
    build_questions,
    construct_single_turn_state,
    describe_outcomes,
    outcomes_from_answers,
    verbalise_outcome,
)
from deepeval.models import DeepEvalBaseLLM, DeepEvalBaseSystemOneModel
from deepeval.models.system_one.schema import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
    SystemOneAnswers,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams, ToolCall


###############################################
# Fakes
###############################################


class FakeSystemOneModel(DeepEvalBaseSystemOneModel):
    """Returns canned answers and records what it was asked."""

    def __init__(self, answers: SystemOneAnswers, cost: Optional[float] = 0.0):
        super().__init__("fake-jev")
        self.answers = answers
        self.cost = cost
        self.calls = []

    def load_model(self, *args, **kwargs):
        return None

    def get_model_name(self):
        return "fake-jev"

    def decide(
        self, state: Any, questions: Dict[str, Any]
    ) -> Tuple[SystemOneAnswers, Optional[float]]:
        self.calls.append((state, questions))
        return self.answers, self.cost

    async def a_decide(
        self, state: Any, questions: Dict[str, Any]
    ) -> Tuple[SystemOneAnswers, Optional[float]]:
        return self.decide(state, questions)


class ExplodingLLM(DeepEvalBaseLLM):
    """Fails if any LLM call is made."""

    def __init__(self):
        super().__init__("exploding-llm")

    def load_model(self, *args, **kwargs):
        return None

    def get_model_name(self):
        return "exploding-llm"

    def generate(self, *args, **kwargs):
        raise AssertionError("LLM must not be called")

    async def a_generate(self, *args, **kwargs):
        raise AssertionError("LLM must not be called")


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


###############################################
# The worked example from the docs: Tool Faithfulness
###############################################

QUESTIONS = [
    Noul(
        "Every fact and figure in actual_output appears in the output of a tool in tools_called.",
        weight=2,
    ),
    Noul(
        "actual_output reports every value returned in tools_called accurately."
    ),
    Score(
        "How much of actual_output is grounded in the outputs in tools_called?",
        levels=[
            "Fabricated",
            "Mostly fabricated",
            "Mostly grounded",
            "Fully grounded",
        ],
    ),
    Choice(
        "What did actual_output do with information the tools did not return?",
        options={
            "left_it_out": 1.0,
            "flagged_it_as_unknown": 1.0,
            "hedged_it": 0.5,
            "stated_it_as_fact": 0.0,
            "nothing_missing": None,
        },
    ),
]

TEST_CASE = LLMTestCase(
    input="What's the weather in Paris right now?",
    actual_output="It's 18°C and sunny in Paris, with a light breeze and around 40% humidity.",
    tools_called=[
        ToolCall(
            name="get_weather",
            input_parameters={"city": "Paris"},
            output={"temp_c": 18, "condition": "sunny"},
        )
    ],
)

EXAMPLE_ANSWERS = SystemOneAnswers(
    nouls={
        "q_0": NoulAnswer(probability=0.30),
        "q_1": NoulAnswer(probability=0.90),
    },
    scores={
        "q_2": ScoreAnswer(
            score=1.7,
            probabilities={0: 0.05, 1: 0.30, 2: 0.55, 3: 0.10},
            confidence=0.55,
        )
    },
    choices={
        "q_3": ChoiceAnswer(
            choice="stated_it_as_fact",
            probabilities={
                "left_it_out": 0.05,
                "flagged_it_as_unknown": 0.05,
                "hedged_it": 0.25,
                "stated_it_as_fact": 0.60,
                "nothing_missing": 0.05,
            },
            confidence=0.6,
        )
    },
)


def make_metric(answers=EXAMPLE_ANSWERS, **kwargs) -> JevEval:
    kwargs.setdefault("include_reason", False)
    kwargs.setdefault("model", ExplodingLLM())
    return JevEval(
        name="Tool Faithfulness",
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.TOOLS_CALLED,
        ],
        questions=QUESTIONS,
        system_one_model=FakeSystemOneModel(answers),
        async_mode=False,
        **kwargs,
    )


###############################################
# Value mapping
###############################################


def test_worked_example_score():
    metric = make_metric()
    score = metric.measure(TEST_CASE)
    # (2*0.30 + 0.90 + 1.7/3 + 0.225/0.95) / 5
    expected = (2 * 0.30 + 0.90 + 1.7 / 3 + 0.225 / 0.95) / 5
    assert score == pytest.approx(expected, abs=1e-9)
    assert score == pytest.approx(0.461, abs=1e-3)
    assert metric.success is False
    assert metric.reason is None
    assert metric.confidence == pytest.approx(0.55)


def test_breakdown_values():
    outcomes = outcomes_from_answers(QUESTIONS, EXAMPLE_ANSWERS)
    assert [o.type for o in outcomes] == ["noul", "noul", "score", "choice"]
    assert outcomes[0].value == pytest.approx(0.30)
    assert outcomes[0].weight == 2
    assert outcomes[0].probabilities == pytest.approx(
        {"true": 0.3, "false": 0.7}
    )
    assert outcomes[1].value == pytest.approx(0.90)
    assert outcomes[2].value == pytest.approx(1.7 / 3)
    assert outcomes[2].probabilities["Mostly grounded"] == pytest.approx(0.55)
    assert outcomes[2].confidence == 0.55
    assert outcomes[3].value == pytest.approx(0.225 / 0.95)
    assert outcomes[3].applicable is True
    assert all(o.applicable for o in outcomes)


def test_choice_excluded_when_not_applicable_mass_dominates():
    answers = EXAMPLE_ANSWERS.model_copy(deep=True)
    answers.choices["q_3"] = ChoiceAnswer(
        choice="nothing_missing",
        probabilities={
            "left_it_out": 0.04,
            "flagged_it_as_unknown": 0.02,
            "hedged_it": 0.02,
            "stated_it_as_fact": 0.02,
            "nothing_missing": 0.90,
        },
        confidence=0.9,
    )
    metric = make_metric(answers)
    score = metric.measure(TEST_CASE)
    outcomes = outcomes_from_answers(QUESTIONS, answers)
    assert outcomes[3].applicable is False
    assert outcomes[3].value is None
    # Only the first three decide the score now.
    assert score == pytest.approx((2 * 0.30 + 0.90 + 1.7 / 3) / 4)
    assert metric.score_breakdown[3]["applicable"] is False


def test_choice_exactly_at_threshold_is_excluded():
    q = Choice("q", options={"a": 1.0, "na": None})
    answers = SystemOneAnswers(
        choices={
            "q_0": ChoiceAnswer(
                choice="na", probabilities={"a": 0.5, "na": 0.5}, confidence=0.0
            )
        }
    )
    outcome = outcomes_from_answers([q], answers)[0]
    assert outcome.applicable is False


def test_choice_non_monotone_credits():
    q = Choice(
        "how did it comply?",
        options={
            "complied_unsafely": 0.0,
            "declined": 0.0,
            "complied_safely": 1.0,
        },
    )
    answers = SystemOneAnswers(
        choices={
            "q_0": ChoiceAnswer(
                choice="complied_safely",
                probabilities={
                    "complied_unsafely": 0.2,
                    "declined": 0.2,
                    "complied_safely": 0.6,
                },
                confidence=0.6,
            )
        }
    )
    assert outcomes_from_answers([q], answers)[0].value == pytest.approx(0.6)


def test_all_excluded_scores_one():
    q = Choice("q", options={"a": 1.0, "na": None})
    answers = SystemOneAnswers(
        choices={
            "q_0": ChoiceAnswer(
                choice="na", probabilities={"a": 0.1, "na": 0.9}, confidence=0.9
            )
        }
    )
    assert aggregate(outcomes_from_answers([q], answers)) == 1.0


def test_weights_change_the_mean():
    a = Noul("a", weight=3)
    b = Noul("b", weight=1)
    answers = SystemOneAnswers(
        nouls={
            "q_0": NoulAnswer(probability=1.0),
            "q_1": NoulAnswer(probability=0.0),
        }
    )
    assert aggregate(outcomes_from_answers([a, b], answers)) == pytest.approx(
        0.75
    )


def test_score_two_levels_maps_to_top_index():
    q = Score("q", levels=["bad", "good"])
    answers = SystemOneAnswers(
        scores={
            "q_0": ScoreAnswer(
                score=0.8, probabilities={0: 0.2, 1: 0.8}, confidence=0.8
            )
        }
    )
    assert outcomes_from_answers([q], answers)[0].value == pytest.approx(0.8)


###############################################
# Request shape
###############################################


def test_state_contains_only_evaluation_params():
    state = construct_single_turn_state(
        [SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT], TEST_CASE
    )
    assert state == {
        "test_case": {
            "input": TEST_CASE.input,
            "actual_output": TEST_CASE.actual_output,
        }
    }


def test_questions_sent_to_jev_have_no_credits():
    questions = build_questions(QUESTIONS)
    assert list(questions) == ["q_0", "q_1", "q_2", "q_3"]
    assert isinstance(questions["q_0"], NoulQuestion)
    assert isinstance(questions["q_2"], ScoreQuestion)
    assert questions["q_2"].levels == QUESTIONS[2].levels
    assert isinstance(questions["q_3"], ChoiceQuestion)
    assert questions["q_3"].options == {
        "left_it_out": None,
        "flagged_it_as_unknown": None,
        "hedged_it": None,
        "stated_it_as_fact": None,
        "nothing_missing": None,
    }


def test_tool_calls_are_structured_in_state():
    state = construct_single_turn_state(
        [SingleTurnParams.TOOLS_CALLED], TEST_CASE
    )
    assert state["test_case"]["tools_called"] == [
        {
            "name": "get_weather",
            "type": "FUNCTION",
            "input_parameters": {"city": "Paris"},
            "output": {"temp_c": 18, "condition": "sunny"},
        }
    ]


def test_one_decide_call_per_measure():
    metric = make_metric()
    metric.measure(TEST_CASE)
    fake: FakeSystemOneModel = metric.system_one_model
    assert len(fake.calls) == 1
    state, questions = fake.calls[0]
    assert set(state["test_case"]) == {
        "input",
        "actual_output",
        "tools_called",
    }
    assert len(questions) == 4


def test_include_reason_false_makes_no_llm_call():
    metric = make_metric(include_reason=False, model=ExplodingLLM())
    metric.measure(TEST_CASE)
    assert metric.reason is None
    assert "canned" not in metric.evaluation_model
    assert metric.evaluation_model == "fake-jev"


def test_include_reason_true_uses_llm_once():
    llm = CannedLLM('{"reason": "The breeze is not in the tool output."}')
    metric = make_metric(include_reason=True, model=llm)
    metric.measure(TEST_CASE)
    assert metric.reason == "The breeze is not in the tool output."
    assert len(llm.prompts) == 1
    assert metric.evaluation_model == "fake-jev + canned-llm"


###############################################
# Reason prompt
###############################################


def test_reason_prompt_covers_every_question_and_hides_numbers():
    llm = CannedLLM('{"reason": "ok"}')
    metric = make_metric(include_reason=True, model=llm)
    metric.measure(TEST_CASE)
    prompt = llm.prompts[0]
    for q in QUESTIONS:
        assert q.text in prompt
    # Verbalised outcomes, not probabilities.
    assert "likely fails" in prompt  # q_0 at 0.30
    assert "clearly holds" in prompt  # q_1 at 0.90
    assert '"Mostly grounded"' in prompt  # 0.55 vs 0.30: no runner-up
    assert '"stated_it_as_fact"' in prompt
    # No probability / score digits leak into the outcomes block. (The test
    # case's own "40%" is content the LLM must be able to cite.)
    outcomes_block = prompt.split("Test Case:")[0]
    assert not re.search(r"\b0\.\d+", outcomes_block)
    assert "%" not in outcomes_block
    # The test case itself is there for the LLM to cite.
    assert TEST_CASE.actual_output in prompt
    assert '"temp_c": 18' in prompt


def test_verbalise_bands():
    def noul(p):
        return outcomes_from_answers(
            [Noul("x")],
            SystemOneAnswers(nouls={"q_0": NoulAnswer(probability=p)}),
        )[0]

    assert verbalise_outcome(noul(0.9)) == "clearly holds"
    assert verbalise_outcome(noul(0.7)) == "likely holds"
    assert verbalise_outcome(noul(0.5)) == "unclear"
    assert verbalise_outcome(noul(0.2)) == "likely fails"
    assert verbalise_outcome(noul(0.05)) == "clearly fails"


def test_describe_outcomes_has_no_probabilities():
    described = describe_outcomes(
        QUESTIONS, outcomes_from_answers(QUESTIONS, EXAMPLE_ANSWERS)
    )
    assert len(described) == 4
    for entry in described:
        assert "probabilities" not in entry
        assert "value" not in entry
        assert "confidence" not in entry
    assert described[2]["levels"] == QUESTIONS[2].levels
    assert described[3]["options"] == list(QUESTIONS[3].options)


###############################################
# Validation
###############################################


def test_empty_questions_rejected():
    with pytest.raises(ValueError):
        JevEval(
            name="x",
            evaluation_params=[SingleTurnParams.INPUT],
            questions=[],
            system_one_model=FakeSystemOneModel(SystemOneAnswers()),
        )


def test_missing_evaluation_params_rejected():
    with pytest.raises(ValueError):
        JevEval(
            name="x",
            questions=[Noul("a")],
            system_one_model=FakeSystemOneModel(SystemOneAnswers()),
        )


def test_wrong_question_type_rejected():
    with pytest.raises(TypeError):
        JevEval(
            name="x",
            evaluation_params=[SingleTurnParams.INPUT],
            questions=["not a question"],
            system_one_model=FakeSystemOneModel(SystemOneAnswers()),
        )


@pytest.mark.parametrize("n", [1, 11])
def test_score_level_count(n):
    with pytest.raises(ValueError):
        Score("q", levels=[f"l{i}" for i in range(n)])


def test_score_levels_unique():
    with pytest.raises(ValueError):
        Score("q", levels=["a", "a"])


def test_choice_credit_out_of_range():
    with pytest.raises(ValueError):
        Choice("q", options={"a": 1.5, "b": 0.0})


def test_choice_all_none_rejected():
    with pytest.raises(ValueError):
        Choice("q", options={"a": None, "b": None})


def test_choice_needs_two_options():
    with pytest.raises(ValueError):
        Choice("q", options={"a": 1.0})


def test_weight_must_be_positive():
    with pytest.raises(ValueError):
        Noul("a", weight=0)


def test_name_suffix():
    metric = make_metric()
    assert metric.__name__ == "Tool Faithfulness [JevEval]"
    metric = make_metric(_include_jev_eval_suffix=False)
    assert metric.__name__ == "Tool Faithfulness"


def test_missing_test_case_param_raises():
    metric = make_metric()
    with pytest.raises(Exception):
        metric.measure(LLMTestCase(input="hi", actual_output="there"))


###############################################
# Live reason (optional)
###############################################


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="needs OPENAI_API_KEY"
)
def test_live_reason_is_grounded():
    metric = make_metric(include_reason=True, model=None)
    metric.measure(TEST_CASE)
    reason = metric.reason.lower()
    # Something from every question shows up.
    assert "breeze" in reason or "humidity" in reason
    assert "sunny" in reason or "18" in reason
    # No probabilities or judge confidence. (The test case's own "40%" may be
    # quoted, so a bare "%" check would be wrong here.)
    assert not re.search(r"\b0\.\d+", reason)
    assert "confiden" not in reason
