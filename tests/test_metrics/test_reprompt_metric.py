import pytest

from deepeval.errors import DeepEvalError
from deepeval.metrics import RepromptMetric
from deepeval.models.system_one.schema import NoulAnswer, SystemOneAnswers
from tests.test_metrics.system_one_fakes import FakeSystemOneModel
from tests.test_metrics.helpers import (
    caller_turn,
    clean_turns,
    measure,
    timed_turn,
    without_place,
)


def _judge(probability, cost=0.002):
    return FakeSystemOneModel(
        answer_fn=lambda questions: SystemOneAnswers(
            nouls={
                key: NoulAnswer(probability=probability) for key in questions
            }
        ),
        cost=cost,
    )


def _silence_then_caller_again():
    return [
        timed_turn("assistant", 2.0, 0.0),
        timed_turn("user", 2.0, 3.0),
        timed_turn("user", 1.0, 9.0, seed=1),
        timed_turn("assistant", 2.0, 11.0, seed=2),
    ]


def test_a_repeat_the_decision_model_confirms_fails_that_line():
    turns = clean_turns()
    turns.insert(3, caller_turn("Hello? Can you hear me?"))
    model = _judge(0.9)
    metric = measure(
        RepromptMetric(use_speech_detection=False, system_one_model=model),
        turns,
    )
    assert metric.score == pytest.approx(3 / 4)
    assert metric.reason == (
        "1 of 4 caller lines had to be repeated before the agent replied."
    )
    assert [without_place(e) for e in metric.score_breakdown["events"]] == [
        {"type": "user_reprompted", "turn": 2, "reprompt_turn": 3}
    ]
    [(state, questions)] = model.calls
    [candidate] = state["candidates"]
    assert candidate["caller_line"] == turns[2].content
    assert candidate["caller_next_line"] == "Hello? Can you hear me?"
    assert candidate["agent_line_before"] == turns[1].content
    assert list(questions) == ["candidate_0"]


def test_a_caller_carrying_on_is_not_a_reprompt():
    turns = clean_turns()
    turns.insert(3, caller_turn("and also my second question."))
    metric = measure(
        RepromptMetric(
            use_speech_detection=False, system_one_model=_judge(0.1)
        ),
        turns,
    )
    assert metric.score == 1.0
    assert metric.score_breakdown["events"] == []


def test_the_decision_model_and_its_cost_are_reported():
    model = _judge(0.9, cost=0.004)
    metric = measure(
        RepromptMetric(use_speech_detection=False, system_one_model=model),
        _silence_then_caller_again(),
    )
    assert metric.evaluation_model == "fake-jev"
    assert metric.evaluation_cost == pytest.approx(0.004)
    [(state, _)] = model.calls
    assert state["candidates"][0]["silence_seconds"] == pytest.approx(4.2)


def test_closing_lines_in_a_row_need_no_reply():
    model = _judge(0.9)
    turns = clean_turns() + [caller_turn("Thanks"), caller_turn("Bye")]
    metric = measure(
        RepromptMetric(use_speech_detection=False, system_one_model=model),
        turns,
    )
    assert metric.score == 1.0
    assert metric.score_breakdown["eligible_turns"] == 3
    assert model.calls == []


def test_lines_never_answered_are_left_to_failed_response():
    model = _judge(0.9)
    turns = clean_turns() + [
        caller_turn("Hello?"),
        caller_turn("Are you there?"),
    ]
    metric = measure(
        RepromptMetric(use_speech_detection=False, system_one_model=model),
        turns,
    )
    assert metric.score == 1.0
    assert metric.score_breakdown["eligible_turns"] == 5
    assert model.calls == []


def test_one_sentence_split_into_turns_is_one_line():
    model = _judge(0.9)
    turns = [
        timed_turn("assistant", 2.0, 0.0),
        timed_turn("user", 2.0, 3.0),
        timed_turn("user", 1.0, 5.5, seed=1),
        timed_turn("assistant", 2.0, 7.5, seed=2),
    ]
    metric = measure(
        RepromptMetric(use_speech_detection=False, system_one_model=model),
        turns,
    )
    assert metric.score == 1.0
    assert model.calls == []


def test_a_pause_mid_thought_is_not_a_reprompt():
    turns = [
        timed_turn("assistant", 2.0, 0.0),
        timed_turn("user", 2.0, 3.0),
        timed_turn("user", 1.0, 7.0, seed=1),
        timed_turn("assistant", 2.0, 9.0, seed=2),
    ]
    metric = measure(RepromptMetric(use_speech_detection=False), turns)
    assert metric.score == 1.0
    assert metric.score_breakdown["eligible_turns"] == 2


def test_speaking_again_after_a_silence_is_a_reprompt():
    metric = measure(
        RepromptMetric(
            use_speech_detection=False, system_one_model=_judge(0.9)
        ),
        _silence_then_caller_again(),
    )
    assert metric.score == pytest.approx(1 / 2)
    assert [without_place(e) for e in metric.score_breakdown["events"]] == [
        {"type": "user_reprompted", "turn": 1, "reprompt_turn": 2}
    ]
    # The silence the caller sat through before saying it again, from where
    # their speech ended.
    [event] = metric.score_breakdown["events"]
    assert event["mark"] == "range"
    assert event["start_time"] == pytest.approx(4.75, abs=0.05)
    assert event["duration_seconds"] == pytest.approx(4.25, abs=0.05)
    assert event["description"] == "Caller repeated themselves after 4.2s"


def test_a_missing_answer_fails_the_metric():
    model = FakeSystemOneModel(answers=SystemOneAnswers())
    with pytest.raises(DeepEvalError, match="no answer"):
        measure(
            RepromptMetric(use_speech_detection=False, system_one_model=model),
            _silence_then_caller_again(),
        )


def test_building_the_metric_needs_no_decision_model_key():
    metric = RepromptMetric()
    assert metric.system_one_model is None
    assert metric.evaluation_model is None
