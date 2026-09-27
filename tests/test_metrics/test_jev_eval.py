"""JevEval: Jev-native score metric. All tests run against a fake System One
model, so no network, API key or LLM is needed: JevEval never calls one."""

import pytest

from deepeval.metrics import JevEval
from deepeval.metrics.jev_eval import Choice, Noul, Score
from deepeval.metrics.jev_eval.utils import (
    aggregate,
    build_questions,
    construct_single_turn_state,
    outcomes_from_answers,
    verbalise_outcome,
)
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
from tests.test_metrics.system_one_fakes import FakeSystemOneModel


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
    # Least decisive answer: the first Noul at P=0.30 -> |2*0.30 - 1| = 0.40,
    # below the Score's 0.55 and the Choice's 0.60.
    assert metric.confidence == pytest.approx(0.40)


def test_breakdown_values():
    outcomes = outcomes_from_answers(QUESTIONS, EXAMPLE_ANSWERS)
    assert [o.type for o in outcomes] == ["noul", "noul", "score", "choice"]
    assert outcomes[0].value == pytest.approx(0.30)
    assert outcomes[0].confidence == pytest.approx(0.40)
    assert outcomes[1].confidence == pytest.approx(0.80)
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
# Strict mode
###############################################


def test_strict_mode_forces_threshold_and_binary_score():
    metric = make_metric(strict_mode=True)
    assert metric.threshold == 1
    score = metric.measure(TEST_CASE)
    # q_0 is at 0.30 (< 0.5) and q_3's chosen option earns 0.0, so it fails.
    assert score == 0.0
    assert metric.success is False
    passed = [o["passed"] for o in metric.score_breakdown]
    assert passed == [False, True, False, False]


def test_strict_mode_perfect_answers_score_one():
    answers = SystemOneAnswers(
        nouls={
            "q_0": NoulAnswer(probability=0.9),
            "q_1": NoulAnswer(probability=0.7),
        },
        scores={
            "q_2": ScoreAnswer(
                score=2.8,
                probabilities={0: 0.0, 1: 0.05, 2: 0.1, 3: 0.85},
                confidence=0.85,
            )
        },
        choices={
            "q_3": ChoiceAnswer(
                choice="left_it_out",
                probabilities={
                    "left_it_out": 0.7,
                    "flagged_it_as_unknown": 0.2,
                    "hedged_it": 0.05,
                    "stated_it_as_fact": 0.03,
                    "nothing_missing": 0.02,
                },
                confidence=0.7,
            )
        },
    )
    metric = make_metric(answers, strict_mode=True)
    assert metric.measure(TEST_CASE) == 1.0
    assert metric.success is True
    assert all(o["passed"] for o in metric.score_breakdown)


def test_strict_mode_skips_not_applicable_choice():
    answers = SystemOneAnswers(
        nouls={
            "q_0": NoulAnswer(probability=0.9),
            "q_1": NoulAnswer(probability=0.9),
        },
        scores={
            "q_2": ScoreAnswer(
                score=2.9,
                probabilities={0: 0.0, 1: 0.0, 2: 0.1, 3: 0.9},
                confidence=0.9,
            )
        },
        choices={
            "q_3": ChoiceAnswer(
                choice="nothing_missing",
                probabilities={
                    "left_it_out": 0.02,
                    "flagged_it_as_unknown": 0.02,
                    "hedged_it": 0.02,
                    "stated_it_as_fact": 0.04,
                    "nothing_missing": 0.90,
                },
                confidence=0.9,
            )
        },
    )
    metric = make_metric(answers, strict_mode=True)
    assert metric.measure(TEST_CASE) == 1.0
    assert metric.score_breakdown[3]["applicable"] is False
    assert metric.score_breakdown[3]["passed"] is None


def test_non_strict_breakdown_has_no_passed_flag():
    metric = make_metric()
    metric.measure(TEST_CASE)
    assert all(o["passed"] is None for o in metric.score_breakdown)


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


def test_include_reason_false_gives_no_reason():
    metric = make_metric(include_reason=False)
    metric.measure(TEST_CASE)
    assert metric.reason is None
    assert metric.evaluation_model == "fake-jev"


def test_model_argument_is_gone():
    # JevEval has no LLM anywhere in its chain, so there is nothing for a
    # `model` argument to configure.
    with pytest.raises(TypeError):
        make_metric(model="gpt-4o")


###############################################
# Reason (deterministic, no LLM)
###############################################


def test_reason_is_deterministic_and_covers_every_question():
    metric = make_metric(include_reason=True)
    metric.measure(TEST_CASE)
    reason = metric.reason
    assert reason.startswith("Decided by fake-jev, minimum confidence 0.40.")
    for q in QUESTIONS:
        assert q.text in reason
    # Verbalised outcomes with the numbers that produced them.
    assert "likely fails (P(yes)=0.30, confidence=0.40, weight=2)" in reason
    assert "clearly holds (P(yes)=0.90, confidence=0.80)" in reason
    assert '"Mostly grounded"' in reason  # 0.55 vs 0.30: no runner-up
    assert "expected level=0.57 of 1.00" in reason
    assert '"stated_it_as_fact" (P=0.60, confidence=0.60)' in reason
    assert reason.endswith(
        "Score: 0.46 (weighted mean of 4 applicable questions)."
    )
    # Same answers, same reason: nothing generated.
    again = make_metric(include_reason=True)
    again.measure(TEST_CASE)
    assert again.reason == reason


def test_reason_marks_strict_results():
    metric = make_metric(include_reason=True, strict_mode=True)
    metric.measure(TEST_CASE)
    assert "strict=fail" in metric.reason
    assert "strict=pass" in metric.reason
    assert metric.reason.endswith(
        "Score: 0.00 (strict mode: at least one applicable question failed)."
    )


def test_reason_marks_not_applicable_choice():
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
    metric = make_metric(answers, include_reason=True)
    metric.measure(TEST_CASE)
    assert "-> not applicable (not applicable" in metric.reason
    assert "weighted mean of 3 applicable questions" in metric.reason


def test_multimodal_test_case_rejected_up_front():
    metric = make_metric()
    with pytest.raises(ValueError, match="text only"):
        metric.measure(
            LLMTestCase(
                input="what is this? [DEEPEVAL:IMAGE:https://x/y.png]",
                actual_output="a cat",
                tools_called=TEST_CASE.tools_called,
                multimodal=True,
            )
        )
    assert metric.system_one_model.calls == []


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
